import glpk
import itertools
import numpy as np
from utils import *
from utils_temporal import *
import xml.etree.ElementTree as ET
import re
from bs4 import BeautifulSoup
"""
relations = ["b", "a", "s", "v", "ii", "i"]
relations_reverse = {"a": "b", "b" : "a", "s" : "s", "ii": "i", "i" : "ii", "v" : "v"}
relations_transitivity = [ [["b"], relations, ["b"], ["i", "v", "b"], ["v", "ii", "b", "s"], ["b", "v", "i"]],
							[relations, ["a"], ["a"], ["i", "v", "a"], ["ii", "a", "s", "v"], ["a", "v", "i"]],
							[["b"], ["a"], ["s"], ["v"], ["ii"], ["i"]],
							[["b", "v", "i"], ["i", "v", "a"], ["v"], relations, ["b", "a", "ii", "v", "s"], ["b", "a", "i", "v", "s"]],
							[["v", "ii", "b", "s"], ["ii", "a", "s", "v"], ["ii"], ["b", "a", "ii", "v", "s"], ["ii"], relations],
							[["b", "v", "i"], ["a", "v", "i"], ["i"], ["b", "a", "i", "v", "s"], relations, ["i"]]]

relations = ["b", "a", "s", "ii", "i"]
relations_transitivity = [ [["b"], relations, ["b"],  ["ii", "b", "s"], ["b",  "i"]],
							[relations, ["a"], ["a"], ["ii", "a", "s"], ["a", "i"]],
							[["b"], ["a"], ["s"],  ["ii"], ["i"]],
							[[ "ii", "b", "s"], ["ii", "a", "s"], ["ii"], ["ii"], relations],
							[["b",  "i"], ["a",  "i"], ["i"], relations, ["i"]]]


relations = ["b", "a", "s", "ii", "i"]
relations_transitivity = [ [["b"], relations, ["b"],  ["ii", "b"], ["b"]],
							[relations, ["a"], ["a"], ["ii", "a"], ["a", "i"]],
							[["b"], ["a"], ["s"],  ["ii"], ["i"]],
							[["b"], [ "a"], ["ii"], ["ii"], relations],
							[["b",  "i"], ["a",  "i"], ["i"], relations, ["i"]]]
"""

relations = ["b", "a", "s", "v"]
relations_reverse = {"a": "b", "b" : "a", "s" : "s", "ii": "i", "i" : "ii", "v" : "v"}
relations_transitivity = [ [["b"], relations, ["b"], ["v", "b"]],
							[relations, ["a"], ["a"], [ "v", "a"]],
							[["b"], ["a"], ["s"], ["v"]],
							[["b", "v"], ["v", "a"], ["v"], relations]]

def get_events_list(dev_predicted, tag_indexer):
	events_to_idx = {}
	relations_softmax = []
	event_pairs = []
	actual_rel = []
	prev_pred = []

	idx_order = []
	for r in relations:
		idx_order.append(tag_indexer.get_index(r))

	for d in dev_predicted:
		if d.e1 not in events_to_idx.keys():
			events_to_idx[d.e1] = len(events_to_idx)
		if d.e2 not in events_to_idx.keys():
			events_to_idx[d.e2] = len(events_to_idx)
		event_pairs.append((events_to_idx[d.e1], events_to_idx[d.e2]))
		relations_softmax.append(d.softmax[idx_order])
		actual_rel.append(d.actual)
		prev_pred.append(d.predicted)
	return event_pairs, events_to_idx, relations_softmax, actual_rel, prev_pred

def get_global_temporal_links(dev_predicted, tag_indexer):
	event_pairs, mapping, scores, actual_rel, prev_pred = get_events_list(dev_predicted, tag_indexer)
	predicted = solve_ilp(event_pairs, scores)

	return zip(predicted, actual_rel, prev_pred)

def check_transitivity(dict_for_checking, transitivity_checks):
	error_idx = []
	for i, check in enumerate(transitivity_checks):
		r1 = dict_for_checking[str(check[0]) + "###" + str(check[1])]
		r2 = dict_for_checking[str(check[1]) + "###" + str(check[2])]
		r3 = dict_for_checking[str(check[0]) + "###" + str(check[2])]

		if r3 not in relations_transitivity[relations.index(r1)][relations.index(r2)]:
			print "WTF is the ILP doing"
			print r1, r2, r3
			error_idx.append(i)
	return error_idx


def get_edge_ids(transitivity_rel, event_pairs):
	e1 = transitivity_rel[0]
	e2 = transitivity_rel[1]
	e3 = transitivity_rel[2]

	return [event_pairs.index((e1,e2)), event_pairs.index((e2,e3)), event_pairs.index((e1,e3))]

def solve_ilp(event_pairs, scores):
		
	distinct_events = []
	for e1,e2 in event_pairs:
		distinct_events.append(e1)
		distinct_events.append(e2)

	distinct_events = list(set(distinct_events))
	transitivity_checks = []
	for i1, i2, i3  in itertools.permutations(distinct_events, 3):
		if (i1, i2) in event_pairs and (i2,i3) in event_pairs and (i1, i3) in event_pairs:
			transitivity_checks.append((i1,i2,i3))

	x = []
	for s in scores:
		x = x + np.ndarray.tolist(s)

	num_relations = len(relations)

	lp = glpk.LPX()
	glpk.env.term_on = False

	lp.cols.add(num_relations * len(event_pairs))
	lp.rows.add(len(event_pairs))

	possible_rel_pairs = [(e1,e2) for e1,e2 in itertools.permutations(relations, 2)]
	possible_rel_pairs = possible_rel_pairs + [(e1, e1) for e1 in relations]

	#lp.rows.add(len(transitivity_checks) * len(possible_rel_pairs)) ## rows for transitivity

	for col in lp.cols:
		col.kind = bool

	lp.obj.maximize = True


	####uniqueness#####
	for row_num, row in enumerate(lp.rows[: len(event_pairs)]):
		cols_temp = range(len(lp.cols))[row_num * len(relations) : row_num * len(relations) + len(relations)]
		row.matrix = [(c, 1.0) for c in cols_temp]
		row.bounds = 1


	####transitivity####
	for idx in range(len(transitivity_checks)):
		transitivity_rel = transitivity_checks[idx]
		edge_ids = get_edge_ids(transitivity_rel, event_pairs)
		
		for r1, r2 in possible_rel_pairs:
			lp.rows.add(1)
			r1_idx = relations.index(r1)
			r2_idx = relations.index(r2)

			col1 = edge_ids[0] * num_relations + r1_idx
			col2 = edge_ids[1] * num_relations + r2_idx
			col3 = [edge_ids[2] * num_relations + relations.index(e) for e in relations_transitivity[r1_idx][r2_idx]]

			matrix = [(col1, 1.0), (col2, 1.0)] + [(c, -1.0) for c in col3]
			lp.rows[-1].matrix = matrix
			lp.rows[-1].bounds = None, 1

	lp.obj[:] = x
	lp.simplex()

	res = np.reshape(lp.cols, (-1, len(relations)))
	#print [ex.primal for ex in lp.rows]
	output_predicted = []
	dict_for_checking = {}
	for r, events in zip(res,event_pairs) :
		max_idx = np.argmax([ex.primal for ex in r])
		max_val = np.max([ex.primal for ex in r])
		output_predicted.append(relations[max_idx])
		dict_for_checking[str(events[0]) + "###" + str(events[1])] = relations[max_idx]

	temp = check_transitivity(dict_for_checking, transitivity_checks)

	return output_predicted

def transitivity_checks_on_gold(data, tag_indexer, word_indexer):
	event_pairs = [(ex.eid1, ex.eid2) for ex in data]
	actual_rel = [ex.label for ex in data]
	sent_pairs = [(ex.sent1, ex.sent2) for ex in data]
	#word_pairs = [(word_indexer.get_object(ex.indexed_words_x[0]), word_indexer.get_object(ex.indexed_words_y[0])) for ex in data]

	distinct_events = []
	for e1,e2 in event_pairs:
		distinct_events.append(e1)
		distinct_events.append(e2)
	distinct_events = list(set(distinct_events))
	
	transitivity_checks = []
	for i1, i2, i3  in itertools.permutations(distinct_events, 3):
		if (i1, i2) in event_pairs and (i2,i3) in event_pairs and (i1, i3) in event_pairs:
			transitivity_checks.append((i1,i2,i3))

	dict_for_checking = {}
	for r, events in zip(actual_rel,event_pairs) :
		dict_for_checking[str(events[0]) + "###" + str(events[1])] = r

	error_idx = check_transitivity(dict_for_checking, transitivity_checks)

	for ex in error_idx:
		e1 = transitivity_checks[ex][0]
		e2 = transitivity_checks[ex][1]
		e3 = transitivity_checks[ex][2]

		idx_1 = event_pairs.index((e1,e2))
		idx_2 = event_pairs.index((e2,e3))
		idx_3 = event_pairs.index((e1,e3))

		r1 = dict_for_checking[str(transitivity_checks[ex][0]) + "###" + str(transitivity_checks[ex][1])]
		r2 = dict_for_checking[str(transitivity_checks[ex][1]) + "###" + str(transitivity_checks[ex][2])]
		r3 = dict_for_checking[str(transitivity_checks[ex][0]) + "###" + str(transitivity_checks[ex][2])]

		print sent_pairs[idx_1], event_pairs[idx_1], r1
		print sent_pairs[idx_2], event_pairs[idx_2], r2
		print sent_pairs[idx_3], event_pairs[idx_3], r3
		print "\n\n\n\n\n"

	print len(error_idx)

def get_tlinks(tlinks_file):
	docs_tlinks = {}
	with open(tlinks_file) as f:
		for line in f.readlines():
			x = line.strip().split("\t")
			doc = x[0]
			if doc in docs_tlinks.keys():
				docs_tlinks[doc].append((x[1],x[2],x[3]))
			else:
				docs_tlinks[doc] = [(x[1],x[2],x[3])]
	return docs_tlinks

def get_ei_e_mapping(documentpath, doc, docs_tlinks_old):
	xmlfile = open(documentpath + doc + ".tml")
	xmldoc = xmlfile.read()
	soup = BeautifulSoup(xmldoc, 'lxml')
	mapping =  [(x.attrs['eventid'], x.attrs['eiid']) for x in soup.findAll('makeinstance')]
	for word in soup.find("timeml"):
		if str(type(word)) == "<class 'bs4.element.Tag'>":
			tagname = word.name
			tag_text = word.text
			tag_attr = word.attrs
			if tagname == "timex3":
				if tag_attr['functionindocument'] == 'CREATION_TIME':
					dct = tag_attr['tid'] 
	docs_tlinks = []
	for d in docs_tlinks_old:
		x = d[0]
		y = d[1]
		z = d[2]

		if x == dct or y == dct:
			continue
		reset = False
		for m in mapping:
			if x == m[0]:
				x = m[1]
				reset = True
			if y == m[0]:
				y = m[1]
				reset = True
			if "t" in x and "t" in y:
				reset = True
		if not reset:
			continue
		
		if "t" in x or "t" in y:
			docs_tlinks.append((x, y, z))
	return docs_tlinks

def find_support(train_links, e1, e2, g):
	pred = None
	e1_support = []
	e2_support = []
	vague = 0
	for d in train_links:
		if d[0] == e1 or d[1] == e1:
			t = d[1] if d[0] == e1 else d[0]
			e1_support.append((d, t))
			if d[2] == "v":
				vague += 1
		if d[0] == e2 or d[1] == e2:
			t = d[1] if d[0] == e2 else d[0]
			e2_support.append((d, t))
			if d[2] == "v":
				vague += 1

	frac = float(vague)/float(len(e1_support) + len(e2_support))

	support_tri = []
	for x in e1_support:
		for y in e2_support:
			for d in train_links:
				if (x[1], y[1]) == (d[0], d[1]):
					support_tri.append((x[0],y[0],d))
				elif (x[1], y[1]) == (d[1], d[0]):
					support_tri.append((x[0],y[0],(d[1], d[0], relations_reverse[d[2]])))

	support_bi = []
	for x in e1_support:
		for y in e2_support:
			if x[1] == y[1]:
				support_bi.append((x[0], y[0]))


	def normalize(s, e):
		if s[0] == e:
			return s
		else:
			return (s[1], s[0], relations_reverse[s[2]])

	

	for (s1,s2) in support_bi:
		s1 = normalize(s1, e1)
		s2 = normalize(s2, e2)

		if s1[2] == "b" and s2[2] == "a":
			pred = "b"
		elif s1[2] == "a" and s2[2] == "b":
			pred =  "a"
		elif s1[2] == "ii" and s2[2] == "a":
			pred = "b"
		elif s1[2] == "ii" and s2[2] == "b":
			pred = "a"
		elif s2[2] == "ii" and s1[2] == "a":
			pred = "a"
		elif s2[2] == "ii" and s1[2] == "b":
			pred = "b"
		elif s2[2] == "s" and s1[2] == "s":
			pred = "s"
		elif s1[2] == "s" and s2[2] == "v":
			pred =  "v"
		elif s1[2] == "v" and s2[2] == "s":
			pred =  "v"
		elif s1[2] == "i" and s2[2] == "a":
			pred =  "b"
		elif s1[2] == "a" and s2[2] == "i":
			pred =  "a"
		elif s1[2] == "i" and s2[2] == "ii":
			pred = "b"
		elif s1[2] == "ii" and s2[2] == "i":
			pred = "a"


	"""

	for (s1, s2, s3) in support_tri:
		s1 = normalize(s1, e1)
		s2 = normalize(s2, e2)

		if s1[2] == "b" and s2[2] == "a" and s3[2] == "b":
			pred = "b"
		elif s1[2] == "a" and s2[2] == "b" and s3[2] == "a":
			pred = "a"
		elif s1[2] == "ii" and s2[2] == "ii" and s3[2] == "a":
			pred = "a"
		elif s1[2] == "ii" and s2[2] == "ii" and s3[2] == "b":
			pred = "b"
		elif s1[2] == "ii" and s2[2] == "a" and s3[2] == "b":
			pred = "b"	

	"""
	return pred, frac










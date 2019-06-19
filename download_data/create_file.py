import os
from bs4 import BeautifulSoup

input_folder = "./ee_files"
f = open("event_pairs_extracted.txt", "w")
for file in os.listdir(input_folder):
	soup = BeautifulSoup(open(os.path.join(input_folder, file)), "html.parser")
	tlinks = [x for x in soup.findAll('tlink')]
	event_pairs = []
	events = []
	for t in tlinks:
		if t.attrs["type"] == "ee":
			event_pairs.append((t.attrs["event1"], t.attrs["event2"], t.attrs["relation"]))
			events.append(t.attrs["event1"])
			events.append(t.attrs["event2"])

	event_sent_dict = {}
	for entry in soup.findAll("entry"):
		if str(type(entry)) == "<class 'bs4.element.Tag'>":
			events_entry = {x.attrs["eiid"]: x.attrs["string"] for x in entry.findAll("event")}
			event_list = set(events).intersection(set(events_entry.keys()))
			if len(event_list) > 0:
				for e in event_list:
					event_sent_dict[e] = (events_entry[e], entry.find("sentence").text)
	
	for e1, e2, r in event_pairs:
		f.write("%s  %s  %s\n" % (event_sent_dict[e1][0], r,  event_sent_dict[e2][0]))
		f.write("%s \n" % event_sent_dict[e1][1])
		f.write("%s \n" % event_sent_dict[e2][1])
		f.write("\n")

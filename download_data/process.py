import os
from bs4 import BeautifulSoup
import re

folders = ["apw_eng"] 
for folder in folders:
	if os.path.exists("./data_processed/" + folder) == False:
		os.mkdir("./data_processed/" + folder)
	for file in os.listdir("./data/" + folder):
		soup = BeautifulSoup(open(os.path.join("./data/" + folder, file)), "html.parser")
		for doc in soup("doc"):
			docname = doc.attrs['id']
			paragraphs = []
			for p in doc("p"):
				para = p.get_text().strip()
				para = re.sub(r"\n+", "\n", para)
				para = para.replace("\n", " ")
				paragraphs.append(para)
			final = " ".join(paragraphs)
			if final.strip() == "":
				continue
			f = open(os.path.join("./data_processed/", folder, docname), "w")
			f.write(final)
			f.close()
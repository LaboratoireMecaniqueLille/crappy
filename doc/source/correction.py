import glob

file_list=glob.glob("./crappy.*.*.rst")

for file_ in file_list:
	try:
		with open(file_) as fin:
			print file_
			lines = fin.readlines()
		temp=lines[0].split("_")[1]
		lines[0]=temp.capitalize()

		with open(file_, 'w') as fout:
			for line in lines:
				fout.write(line)
	except:
		pass
			
			
file_list_blocks=glob.glob("./crappy.blocks.*.rst")
modify_list=set(file_list)-set(file_list_blocks)-set(['./crappy.sensor._ximeaSensor.rst'])-set(['./crappy.sensor._jaiSensor.rst'])
for file_ in modify_list:
	try:
		with open(file_, 'r') as fout:
			print file_
			lines = fout.readlines()
			for i,line in enumerate(lines):
				print line
				if "show-inheritance" in line:
					print "line deleted"
					lines.pop(i)
					
		with open(file_, 'w') as fout:
			for line in lines:
				fout.write(line)
				
	except Exception as e:
		print "exception : ", e
		pass

file_list_blocks_links=glob.glob("./crappy.blocks.*.rst")+glob.glob("./crappy.links.*.rst")
for file_ in file_list_blocks_links:
	try:
		with open(file_, 'r') as fout:
			print file_
			lines = fout.readlines()
			for i,line in enumerate(lines):
				print line
				if "undoc-members" in line:
					print "line deleted"
					lines.pop(i)
					
		with open(file_, 'w') as fout:
			for line in lines:
				fout.write(line)
				
	except Exception as e:
		print "exception : ", e
		pass
from sys import argv

if len(argv) != 3:
  print("Usage: python create_indel_locations <chromosome_number> <include_filtered [0 or 1]>")
  exit()

chromosome_id = argv[1]
include_f = int(argv[2])

filename = "/home/cs273b_home/project_data/gnomad_indels.tsv"

locations = []
with open(filename) as f:
  for i, line in enumerate(f):
    if i % 1000000 == 0: print("Line {}...".format(i))
    if i == 0: continue
    l = line.split()
    chromosome = l[0]
    location = int(l[1]) # The position is 1-indexed! We should do - 1 here?
    if chromosome == chromosome_id and (include_f or l[4] != 'true'):
      locations.append(location)

with open("indelLocations" + chromosome_id + ".txt", 'w') as f:
  for l in locations:
    f.write(str(l) + '\n')

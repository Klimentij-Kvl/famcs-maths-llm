retrieved = ['a', 'b', 'c', 'd', 'e']
reference = ['b']

print(sum([s in retrieved for s in reference]) / len(reference))
print(sum([s in retrieved for s in reference]) / len(retrieved))
import json

with open("UAV20L.json","r") as f:
	data = json.load(f)
print(type(data))
print (data.keys())
print (type(data['person7']))
print (data['person7'].keys())
print (data['person7']['video_dir'])
print (data['person7']['init_rect'])
print (data['person7']['img_names'][:3])
print (data['person7']['gt_rect'][:3])
print (data['person7']['attr'])
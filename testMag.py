import os, time
p = "/Volumes/data/Good imaging data/Multi-phenotype training"
t0 = time.time()
e = os.listdir(p)
print(len(e), "entries in", round(time.time()-t0, 1), "s")

skip = {'processedimages','processed_images_py','numerical_data_py','numericaldata','plots','__pycache__','.git','checkpoints'}
file_exts = {'tif','tiff','csv','json','xlsx','xls','pdf','png','jpg','mp4','npz','npy','log','txt','py','r','md'}

candidates = []
for name in e:
    if name.startswith('.') or name.startswith('~$'):
        continue
    if name.lower() in skip:
        continue
    if '.' in name and name.rsplit('.',1)[1].lower() in file_exts:
        continue
    candidates.append(name)

print(len(candidates), "candidates in", round(time.time()-t0, 1), "s")
for c in candidates[:5]:
    print(" ", c)

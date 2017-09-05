n_objects = []
sizes = []
delta = range(1,10)
masks = []
for d in delta :
    mser_delta = d
    mser_min = 20    # min size of object
    mser_max = 4000  # max size of object

    mser.setDelta(mser_delta)
    mser.setMinArea(mser_min)
    mser.setMaxArea(mser_max)
    regions = mser.detectRegions(I, None)
    
    n = len(regions)
    areas = [len(r) for r in regions]
    avg_size = np.median(areas)
    
    n_objects.append(n)
    sizes.append(avg_size)
    
    masks.append(regions2mask(regions, I.shape))
    
plt.figure(figsize=(15,5))
plt.subplot(131)
plt.plot(delta, n_objects)
plt.xlabel('delta')
plt.ylabel('# objects')
plt.subplot(132)
plt.plot(delta, sizes)
plt.xlabel('delta')
plt.ylabel('avg. size')
plt.subplot(133)
plt.scatter(n_objects, sizes, c=delta)
plt.xlabel('# objects')
plt.ylabel('avg. size')

plt.show()
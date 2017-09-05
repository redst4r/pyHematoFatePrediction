tmin, tmax = 3300,3400

plt.figure()
for pos in [54,55]:
    prop_generator = (movie.get_segmented_objects_from_images(timepoint=_, position=pos)[0] 
                      for _ in range(tmin,tmax))
    Q = toolz.concat(prop_generator)
    timepoints = toolz.pluck('timepoint', Q)
    timepoint_histogram = toolz.frequencies(timepoints)
    t, freq = list(zip(*timepoint_histogram.items()))
    plt.plot(t,freq)
    plt.xlabel('Time')
    plt.ylabel('#cells')
plt.show()
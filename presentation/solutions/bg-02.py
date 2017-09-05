# Warning: this loads all images into memory, calculating the average once all are loaded! Memory issues!!
Q = [load_raw_image(position=pos, timepoint=t) for t in movie.get_timepoints(position=pos)]
meanI= np.stack(Q).mean(0)

# # smarter: 
# import toolz
# time_iter = movie.get_timepoints(position=pos)
# sumI = toolz.reduce(lambda acc, t: acc+load_raw_image(position=pos, timepoint=t), time_iter, np.zeros((1040, 1388)))
# meanI = sumI/len(time_iter)
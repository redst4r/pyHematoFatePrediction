pos=54
average_intensity_over_time = [np.mean(load_raw_image(position=pos, timepoint=t)) 
                               for t in movie.get_timepoints(position=pos)]
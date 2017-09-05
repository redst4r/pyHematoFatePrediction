pos, time = 54, 3310
Iobs = load_raw_image(pos, time)
b = average_intensity_over_time[movie.get_timepoints(pos).index(time)]

Inorm = Iobs/meanI/b
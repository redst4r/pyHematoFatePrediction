aggr_l = aggregate_cell_scores(df)
aggr_l.head(10)

get_auc(df['score1'], df['y1'], do_plot=True)  # without aggregation
get_auc(aggr_l['softvote1'], aggr_l['y1'], do_plot=True);
get_auc(aggr_l['hardvote1'], aggr_l['y1'], do_plot=True);
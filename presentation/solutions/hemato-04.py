auc_list = []  
n_cells = []

inv_gen_range = range(-5,0)  # split into the different generations relative to onset
for g in inv_gen_range:
    df_generation = aggr_l.query('gens==@g')
    auc = get_auc(df_generation['softvote1'], df_generation['y1'])
    auc_list.append(auc)
    n_cells.append(len(df_generation))

plt.figure(figsize=(10,3))

plt.subplot(121)
plt.plot(inv_gen_range, auc_list, marker='x')
plt.xlabel('inverted generation');
plt.ylabel('AUC');

plt.subplot(122)
plt.plot(inv_gen_range, n_cells, marker='x');
plt.xlabel('inverted generation');
plt.ylabel('number of cells');
import pandas as pd
import glob
import os


def merge_seeds(file_name_pattern='CUB_results_s*_hessian_exact.csv', data_dir='./logs/ISR_Hessian_results_new'):
    # Create the full pattern for glob
    full_pattern = os.path.join(data_dir, file_name_pattern)

    # Use glob to get all file paths matching the pattern
    all_files = glob.glob(full_pattern)

    # Initialize an empty list to hold the dataframes
    df_list = []

    # Loop through the files, read each into a dataframe, and append it to the list
    for file_path in all_files:
        df = pd.read_csv(file_path)
        df['gradient_alpha'] = df['gradient_alpha'].astype(float).round(8)
        df['hessian_beta'] = df['hessian_beta'].astype(float).round(8)
        df_list.append(df)

    # Concatenate all the dataframes in the list
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df['gradient_alpha'] = pd.to_numeric(merged_df['gradient_alpha'], errors='coerce')
    merged_df['hessian_beta'] = pd.to_numeric(merged_df['hessian_beta'], errors='coerce')
    merged_df['penalty_anneal_iters'] = pd.to_numeric(merged_df['penalty_anneal_iters'], errors='coerce')
    # Define columns for the groupby operation based on 'fishr' in filename

    if 'fishr' in file_name_pattern:
        group_columns = ['dataset', 'split', 'method', 'ISR_class', 'ISR_scale', 'num_iter', 'ema', 'lambda',
                         'penalty_anneal_iters']
    elif 'coral' in file_name_pattern:
        group_columns = ['dataset', 'split', 'method', 'ISR_class', 'ISR_scale', 'num_iter', 'mmd_gamma']
    else:
        group_columns = ['dataset', 'split', 'method', 'ISR_class', 'ISR_scale', 'num_iter', 'gradient_alpha',
                     'hessian_beta', 'penalty_anneal_iters']
    # Add 'num_runs' before groupby
    merged_df['num_runs'] = merged_df.groupby(group_columns).transform('size')
    # merged_df['seed_runs'] = merged_df.groupby(group_columns)['seed'].transform(lambda x: list(x.unique()))
    seed_runs = merged_df.groupby(group_columns)['seed'].agg(lambda x: list(set(x))).rename('seed_runs')

    # Merge this aggregated data back to the original DataFrame if needed
    merged_df = merged_df.merge(seed_runs, on=group_columns, how='left')

    # Group and aggregate data

    merged_df['num_iter'] = pd.to_numeric(merged_df['num_iter'], errors='coerce')


    # base_columns = ['dataset', 'seed', 'split', 'method', 'ISR_class', 'ISR_scale', 'num_iter', 'acc-0', 'acc-1', 'acc-2', 'acc-3', 'worst_group', 'avg_acc', 'worst_acc','num_runs','seed_runs']
    if 'fishr' in file_name_pattern:
        merged_df = merged_df[['dataset','seed','split','method','ISR_class','ISR_scale','num_iter', 'ema','lambda','penalty_anneal_iters', 'acc-0', 'acc-1', 'acc-2', 'acc-3', 'worst_group', 'avg_acc', 'worst_acc','num_runs','seed_runs']]

    elif 'coral' in file_name_pattern:
        merged_df = merged_df[['dataset','seed','split','method','ISR_class','ISR_scale','num_iter', 'mmd_gamma', 'acc-0', 'acc-1', 'acc-2', 'acc-3', 'worst_group', 'avg_acc', 'worst_acc','num_runs','seed_runs']]
    else:
        merged_df = merged_df[['dataset','seed','split','method','ISR_class','ISR_scale','num_iter', 'gradient_alpha', 'hessian_beta','penalty_anneal_iters', 'acc-0', 'acc-1', 'acc-2', 'acc-3', 'worst_group', 'avg_acc', 'worst_acc','num_runs','seed_runs']]


    if 'CUB' in file_name_pattern:
        merged_df = merged_df[merged_df['num_iter']==300]
    elif 'CelebA' in file_name_pattern:
        merged_df = merged_df[merged_df['num_iter']==50]
    elif 'MultiNLI' in file_name_pattern:
        merged_df = merged_df[merged_df['num_iter']==3]

    merged_df['avg_acc'] = pd.to_numeric(merged_df['avg_acc'], errors='coerce')
    merged_df['worst_acc'] = pd.to_numeric(merged_df['worst_acc'], errors='coerce')
    grouped = merged_df.groupby(group_columns).agg({
        'avg_acc': ['mean', 'sem'],
        'worst_acc': ['mean', 'sem'],
        'num_runs': 'first',
        'seed_runs': 'first'
    })

    # Renaming the columns for clarity
    grouped.columns = ['avg_acc_mean', 'avg_acc_sem', 'worst_acc_mean', 'worst_acc_sem', 'num_runs','seed_runs']
    # Resetting the index if you want the grouped columns back as regular columns
    grouped = grouped.reset_index()

    # cleaned_grouped = grouped.dropna(subset=['avg_acc_sem', 'worst_acc_sem'])
    cleaned_grouped = grouped
    # Display the cleaned DataFrame
    # print(cleaned_grouped)
    val = cleaned_grouped[cleaned_grouped['split'] == 'val']
    test = cleaned_grouped[cleaned_grouped['split'] == 'test']
    num_runs = len(all_files)
    dataset = file_name_pattern.split('_')[0]
    if 'fishr' in file_name_pattern:
        val.to_csv(f'{data_dir}/{dataset}_{num_runs}runs_fishr_val.csv', index=False)
        test.to_csv(f'{data_dir}/{dataset}_{num_runs}runs_fishr_test.csv', index=False)
    elif 'coral' in file_name_pattern:
        val.to_csv(f'{data_dir}/{dataset}_{num_runs}runs_coral_val.csv', index=False)
        test.to_csv(f'{data_dir}/{dataset}_{num_runs}runs_coral_test.csv', index=False)
    elif 'hgp' in file_name_pattern:
        val.to_csv(f'{data_dir}/{dataset}_{num_runs}runs_hgp_val.csv', index=False)
        test.to_csv(f'{data_dir}/{dataset}_{num_runs}runs_hgp_test.csv', index=False)
    else:
        val.to_csv(f'{data_dir}/{dataset}_{num_runs}runs_val.csv', index=False)
        test.to_csv(f'{data_dir}/{dataset}_{num_runs}runs_test.csv', index=False)
    # print(f"Saved {dataset}_{num_runs}runs_val.csv and {dataset}_{num_runs}runs_test.csv in {data_dir}")
    return val, test

def find_best_isr(data_dir='./logs/ISR_hessian_results_new', file_name='CUB_5runs_val.csv', worst_case = False):
    val = pd.read_csv(os.path.join(data_dir, file_name))
    test = pd.read_csv(os.path.join(data_dir, file_name.replace('val', 'test')))
    val = val[val['gradient_alpha'] == 0]
    test = test[test['gradient_alpha'] == 0]
    val = val[val['hessian_beta'] == 0]
    test = test[test['hessian_beta'] == 0]
    val = val[val['penalty_anneal_iters'] == 0]
    test = test[test['penalty_anneal_iters'] == 0]

    best_hyperparameters, best_test_performance = find_best_hps(val, test, worst_case)

    return best_hyperparameters, best_test_performance

def find_best_gm(data_dir='./logs/ISR_hessian_results_new', file_name='CUB_5runs_val.csv', worst_case = False):
    val = pd.read_csv(os.path.join(data_dir, file_name))
    test = pd.read_csv(os.path.join(data_dir, file_name.replace('val', 'test')))
    val = val[val['hessian_beta'] == 0]
    test = test[test['hessian_beta'] == 0]
    val = val[val['gradient_alpha'] != 0]
    test = test[test['gradient_alpha'] != 0]
    best_hyperparameters, best_test_performance = find_best_hps(val, test, worst_case)

    return best_hyperparameters, best_test_performance

def find_best_hm(data_dir='./logs/ISR_hessian_results_new', file_name='CUB_5runs_val.csv', worst_case = False):
    val = pd.read_csv(os.path.join(data_dir, file_name))
    test = pd.read_csv(os.path.join(data_dir, file_name.replace('val', 'test')))
    val = val[val['gradient_alpha'] == 0]
    test = test[test['gradient_alpha'] == 0]
    val = val[val['hessian_beta'] != 0]
    test = test[test['hessian_beta'] != 0]
    best_hyperparameters, best_test_performance = find_best_hps(val, test, worst_case)

    return best_hyperparameters, best_test_performance

def find_best_gm_hm(data_dir='./logs/ISR_Hessian_results_new', file_name='CUB_5runs_val.csv', worst_case = False):
    val = pd.read_csv(os.path.join(data_dir, file_name))
    test = pd.read_csv(os.path.join(data_dir, file_name.replace('val', 'test')))
    val = val[val['gradient_alpha'] != 0]
    test = test[test['gradient_alpha'] != 0]
    val = val[val['hessian_beta'] != 0]
    test = test[test['hessian_beta'] != 0]
    val = val[val['num_runs'] >= 5]
    test = test[test['num_runs'] >= 5]
    best_hyperparameters, best_test_performance = find_best_hps(val, test, worst_case)
    if best_test_performance['num_runs'].iloc[0] < 5:
        print("Not enough runs")
        return best_hyperparameters, best_test_performance

    return best_hyperparameters, best_test_performance

def find_best_hps(val_df, test_df, worst_case = False):
    val_df = val_df[val_df['ISR_scale'] == 0]
    test_df = test_df[test_df['ISR_scale'] == 0]
    # val_df = val_df[val_df['avg_acc_sem'].isna() == False]
    # test_df = test_df[test_df['avg_acc_sem'].isna() == False]
    col_mean = 'worst_acc_mean' if worst_case else 'avg_acc_mean'
    col_sem = 'worst_acc_sem' if worst_case else 'avg_acc_sem'
    # Step 1: Identify the highest average accuracy
    max_avg_acc_mean = val_df[col_mean].max()

    # Filter to include only those with the highest average accuracy
    top_performers = val_df[val_df[col_mean] == max_avg_acc_mean]

    # Step 2: Select based on Standard Error
    # Identify the entry with the smallest standard error among the top performers
    if len(top_performers) > 1:
        best_val_hyperparameters = top_performers.loc[top_performers[col_sem].idxmin()]
    else:
        best_val_hyperparameters = top_performers.iloc[0]

    # Display the chosen hyperparameters

    # print("Chosen hyperparameters for best",f"{'worst case accuracy' if worst_case else 'average accuracy'}"  ":\n", best_val_hyperparameters)

    # Extracting hyperparameters
    best_hyperparameters = {
        'ISR_class': best_val_hyperparameters['ISR_class'],
        'ISR_scale': best_val_hyperparameters['ISR_scale'],
        'num_iter': best_val_hyperparameters['num_iter'],
        'gradient_alpha': best_val_hyperparameters['gradient_alpha'],
        'hessian_beta': best_val_hyperparameters['hessian_beta'],
        'penalty_anneal_iters': 0 if 'penalty_anneal_iters' not in best_val_hyperparameters else best_val_hyperparameters['penalty_anneal_iters'],
        'num_runs': best_val_hyperparameters['num_runs']
    }

    # Filter the test set for these hyperparameters
    best_test_performance = test_df[
        (test_df['ISR_class'] == best_hyperparameters['ISR_class']) &
        (test_df['ISR_scale'] == best_hyperparameters['ISR_scale']) &
        (test_df['num_iter'] == best_hyperparameters['num_iter']) &
        (test_df['gradient_alpha'] == best_hyperparameters['gradient_alpha']) &
        (test_df['hessian_beta'] == best_hyperparameters['hessian_beta']) &
        (test_df['penalty_anneal_iters'] == best_hyperparameters['penalty_anneal_iters']) &
        (test_df['num_runs'] == best_hyperparameters['num_runs'])
    ]
    pd.set_option('display.max_columns', None)

    # Display the performance on the test set
    print("Performance on test for best",f"{'worst case accuracy' if worst_case else 'average accuracy'}"  ":\n",
          best_test_performance[['dataset', 'split','gradient_alpha','hessian_beta','penalty_anneal_iters', 'avg_acc_mean', 'avg_acc_sem', 'worst_acc_mean', 'worst_acc_sem','num_runs']])

    return best_hyperparameters, best_test_performance


def find_best_coral(data_dir='./logs/ISR_hessian_results_new', file_name='CUB_5runs_coral_val.csv', worst_case = False):
    val = pd.read_csv(os.path.join(data_dir, file_name))
    test = pd.read_csv(os.path.join(data_dir, file_name.replace('val', 'test')))
    val= val[val['num_runs'] == 5]
    test= test[test['num_runs'] == 5]
    val_df = val[val['ISR_scale'] == 0]
    test_df = test[test['ISR_scale'] == 0]

    col_mean = 'worst_acc_mean' if worst_case else 'avg_acc_mean'
    col_sem = 'worst_acc_sem' if worst_case else 'avg_acc_sem'
    # Step 1: Identify the highest average accuracy
    max_avg_acc_mean = val_df[col_mean].max()

    # Filter to include only those with the highest average accuracy
    top_performers = val_df[val_df[col_mean] == max_avg_acc_mean]

    # Step 2: Select based on Standard Error
    # Identify the entry with the smallest standard error among the top performers
    if len(top_performers) > 1:
        best_val_hyperparameters = top_performers.loc[top_performers[col_sem].idxmin()]
    else:
        best_val_hyperparameters = top_performers.iloc[0]

    # Display the chosen hyperparameters

    # print("Chosen hyperparameters for best", f"{'worst case accuracy' if worst_case else 'average accuracy'}"  ":\n",
    #       best_val_hyperparameters)

    # Extracting hyperparameters
    best_hyperparameters = {
        'Dataset': best_val_hyperparameters['dataset'],
        'ISR_class': best_val_hyperparameters['ISR_class'],
        'ISR_scale': best_val_hyperparameters['ISR_scale'],
        'num_iter': best_val_hyperparameters['num_iter'],
        'num_runs': best_val_hyperparameters['num_runs'],
        'mmd_gamma': best_val_hyperparameters['mmd_gamma'],
    }

    # Filter the test set for these hyperparameters
    best_test_performance = test_df[
        (test_df['ISR_class'] == best_hyperparameters['ISR_class']) &
        (test_df['ISR_scale'] == best_hyperparameters['ISR_scale']) &
        (test_df['num_iter'] == best_hyperparameters['num_iter']) &

        (test_df['mmd_gamma'] == best_hyperparameters['mmd_gamma'])
    ]
    pd.set_option('display.max_columns', None)
    print(best_hyperparameters)
    # Display the performance on the test set
    print("Performance on test for best", f"{'worst case accuracy' if worst_case else 'average accuracy'}"  ":\n",
          best_test_performance[
              ['dataset', 'split', 'mmd_gamma', 'avg_acc_mean', 'avg_acc_sem', 'worst_acc_mean',
               'worst_acc_sem']])
    return best_hyperparameters, best_test_performance



def find_best_fishr(data_dir='./logs/ISR_hessian_results_new', file_name='CUB_5runs_fishr_val.csv', worst_case = False):
    val = pd.read_csv(os.path.join(data_dir, file_name))
    test = pd.read_csv(os.path.join(data_dir, file_name.replace('val', 'test')))
    val= val[val['num_runs'] == 5]
    test= test[test['num_runs'] == 5]
    val_df = val[val['ISR_scale'] == 0]
    test_df = test[test['ISR_scale'] == 0]

    col_mean = 'worst_acc_mean' if worst_case else 'avg_acc_mean'
    col_sem = 'worst_acc_sem' if worst_case else 'avg_acc_sem'
    # Step 1: Identify the highest average accuracy
    max_avg_acc_mean = val_df[col_mean].max()

    # Filter to include only those with the highest average accuracy
    top_performers = val_df[val_df[col_mean] == max_avg_acc_mean]

    # Step 2: Select based on Standard Error
    # Identify the entry with the smallest standard error among the top performers
    if len(top_performers) > 1:
        best_val_hyperparameters = top_performers.loc[top_performers[col_sem].idxmin()]
    else:
        best_val_hyperparameters = top_performers.iloc[0]

    # Display the chosen hyperparameters

    # print("Chosen hyperparameters for best", f"{'worst case accuracy' if worst_case else 'average accuracy'}"  ":\n",
    #       best_val_hyperparameters)

    # Extracting hyperparameters
    best_hyperparameters = {
        'Dataset': best_val_hyperparameters['dataset'],
        'ISR_class': best_val_hyperparameters['ISR_class'],
        'ISR_scale': best_val_hyperparameters['ISR_scale'],
        'num_iter': best_val_hyperparameters['num_iter'],
        'ema': best_val_hyperparameters['ema'],
        'lambda': best_val_hyperparameters['lambda'],
        'penalty_anneal_iters': best_val_hyperparameters['penalty_anneal_iters'],
        'num_runs': best_val_hyperparameters['num_runs']
    }

    # Filter the test set for these hyperparameters
    best_test_performance = test_df[
        (test_df['ISR_class'] == best_hyperparameters['ISR_class']) &
        (test_df['ISR_scale'] == best_hyperparameters['ISR_scale']) &
        (test_df['num_iter'] == best_hyperparameters['num_iter']) &
        (test_df['ema'] == best_hyperparameters['ema']) &
        (test_df['lambda'] == best_hyperparameters['lambda']) &
        (test_df['penalty_anneal_iters'] == best_hyperparameters['penalty_anneal_iters'])
    ]
    pd.set_option('display.max_columns', None)
    print(best_hyperparameters)
    # Display the performance on the test set
    if 'fishr' in file_name:
        print("Performance on test for best", f"{'worst case accuracy' if worst_case else 'average accuracy'}"  ":\n",
              best_test_performance[
                  ['dataset', 'split', 'ema','lambda','penalty_anneal_iters', 'avg_acc_mean', 'avg_acc_sem', 'worst_acc_mean',
                   'worst_acc_sem', 'num_runs']])
    else:
        print("Performance on test for best", f"{'worst case accuracy' if worst_case else 'average accuracy'}"  ":\n",
              best_test_performance[
                  ['dataset', 'split', 'gradient_alpha', 'hessian_beta', 'avg_acc_mean', 'avg_acc_sem', 'worst_acc_mean',
                   'worst_acc_sem']])
    return best_hyperparameters, best_test_performance


# def write_latex_table(df, filename):
    # Construct LaTeX table with specific formatting and structure
    #  header = r'''\begin{table*}[ht!]
    # \centering
    # \resizebox{.99\textwidth}{!}{%
    # \centering
    # \setlength{\tabcolsep}{4.5pt}
    # \begin{tabular}{@{}ccccccccccc@{}}
    # \toprule
    # \multicolumn{1}{c}{\textbf{Dataset}} &
    # \multicolumn{1}{c}{\textbf{Backbone}} &
    # \multicolumn{3}{c}{\textbf{Average Accuracy}} &
    # \multicolumn{3}{c}{\textbf{Worst-Group Accuracy}} \\
    # \cmidrule(l){3-5} \cmidrule(l){6-8}
    # & \multicolumn{1}{c}{\text{~}}&\textbf{ERM}& \textbf{Fishr}& \textbf{Hessian} & \textbf{ERM}& \textbf{Fishr}& \textbf{Hessian}\\
    # \midrule
    # '''
    #     footer = r''' \\
    # \bottomrule
    # \end{tabular}}
    # \caption{Test accuracy(\%) with standard error over three datasets.}
    # \label{tab:real-datasets}
    # \end{table*}'''
    #
    #
    #
    # # Convert DataFrame to LaTeX table format
    # # body = df.to_latex(index=False, header=False, escape=False, column_format="lccccccc")
    # latex_rows = []
    # for index, row in df.iterrows():
    #     latex_row = f"""
    #     \\multicolumn{{1}}{{c}}{{{row['Dataset']}}} & {row['Backbone']} &
    #     {row['ERM_avg']} & {row['Fishr_avg']} & {row['Hessian_avg']} &
    #     {row['ERM_worst']} & {row['Fishr_worst']} & {row['Hessian_worst']}"""
    #     latex_rows.append(latex_row)
    #
    # table_body = " \\\\\n\midrule".join(latex_rows)
    # full_latex = header + table_body + footer
    #
    # # Write to .tex file
    # if not filename.endswith('.tex'):
    #     filename += '.tex'
    # if not os.path.exists(os.path.dirname(filename)):
    #     print("Wrong path")
    # with open(filename, 'w') as f:
    #     f.write(full_latex)
    #
    # print(f"LaTeX table written to {filename}")

def write_latex_table(df, filename):
        # Construct LaTeX table with specific formatting and structure
        header = r'''
\begin{table}[ht!]
    \centering
    \caption{Test accuracy (\%) with standard error over three datasets. In the parentheses are the backbone models for each dataset. Each experiment is repeated over 5 random seeds.}
    \centering
    \resizebox{\textwidth}{!}{%
        \begin{tabular}{@{}lcccccc@{}}
            \toprule
            \textbf{Method} & \multicolumn{2}{c}{\textbf{Waterbirds (CLIP ViT-B/32)}} & \multicolumn{2}{c}{\textbf{CelebA (CLIP ViT-B/32)}} & \multicolumn{2}{c}{\textbf{MultiNLI (BERT)}}                                                                         \\
            \cmidrule(l){2-3} \cmidrule(l){4-5} \cmidrule(l){6-7}
                            & \textbf{Average}                                        & \textbf{Worst-Group}                                & \textbf{Average}                             & \textbf{Worst-Group}  & \textbf{Average}      & \textbf{Worst-Group}  \\
            \midrule
        '''
        footer = r'''
        \bottomrule
    \end{tabular}\label{tab:real-datasets-transposed}}
\end{table}'''

        # Ensure DataFrame is in the correct format
        required_columns = [
            'Dataset', 'Backbone', 'ERM_avg', 'Fishr_avg', 'Coral_avg', 'Hessian_avg',
            'ERM_worst', 'Fishr_worst', 'Coral_worst', 'Hessian_worst'
        ]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        methods = ['ERM', 'Fishr', 'CORAL', 'HGP', 'CMA']

        # Create a dictionary to hold data for each method
        method_data = {method: {dataset: {} for dataset in df['Dataset'].unique()} for method in methods}

        for _, row in df.iterrows():
            dataset = row['Dataset']
            method_data['ERM'][dataset]['Average'] = row['ERM_avg']
            method_data['ERM'][dataset]['Worst-Group'] = row['ERM_worst']
            method_data['Fishr'][dataset]['Average'] = row['Fishr_avg']
            method_data['Fishr'][dataset]['Worst-Group'] = row['Fishr_worst']
            method_data['CORAL'][dataset]['Average'] = row['Coral_avg']
            method_data['CORAL'][dataset]['Worst-Group'] = row['Coral_worst']
            method_data['HGP'][dataset]['Average'] = row['HGP_avg']
            method_data['HGP'][dataset]['Worst-Group'] = row['HGP_worst']
            method_data['CMA'][dataset]['Average'] = row['Hessian_avg']
            method_data['CMA'][dataset]['Worst-Group'] = row['Hessian_worst']

        # Construct the LaTeX table body
        latex_rows = []
        for method in methods:
            row = [method]
            for dataset in df['Dataset'].unique():
                row.append(method_data[method][dataset]['Average'])
                row.append(method_data[method][dataset]['Worst-Group'])
            latex_rows.append(" & ".join(row) + " \\\\")
        table_body = " \n            ".join(latex_rows)

        full_latex = header + table_body + footer

        # Write to .tex file
        if not filename.endswith('.tex'):
            filename += '.tex'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            f.write(full_latex)

        print(f"LaTeX table written to {filename}")

def main():
    cub_pattern = 'CUB_results_s*_hessian_exact.csv'
    cub_fishr = 'CUB_results_s*_fishr.csv'
    cub_coral = 'CUB_results_s*_coral.csv'
    cub_hgp = 'CUB_results_s*_hessian_hgp.csv'

    celeba_pattern = 'CelebA_results_s*_hessian_exact.csv'
    celeba_fishr = 'CelebA_results_s*_fishr.csv'
    celeba_coral = 'CelebA_results_s*_coral.csv'
    celeba_hgp = 'CelebA_results_s*_hessian_hgp.csv'

    multiNLI_pattern = 'MultiNLI_results_s*_hessian_exact.csv'
    multiNLI_fishr = 'MultiNLI_results_s*_fishr.csv'
    multiNLI_coral = 'MultiNLI_results_s*_coral.csv'
    multiNLI_hgp = 'MultiNLI_results_s*_hessian_hgp.csv'
    data_dir= './logs/ISR_Hessian_results_new'


    for pattern in [cub_pattern, celeba_pattern, multiNLI_pattern, cub_fishr, celeba_fishr, multiNLI_fishr, cub_coral, celeba_coral, multiNLI_coral, cub_hgp, celeba_hgp, multiNLI_hgp]:
        merge_seeds(file_name_pattern=pattern, data_dir=data_dir)


    worst_case = True  # Set this according to your need to get worst case or average case accuracies

    datasets = {
        'Waterbirds': {'file_name': 'CUB_5runs_val.csv', 'backbone': 'CLIP (ViT-B/32)'},
        'CelebA': {'file_name': 'CelebA_5runs_val.csv', 'backbone': 'CLIP (ViT-B/32)'},
        'MultiNLI': {'file_name': 'MultiNLI_5runs_val.csv', 'backbone': 'BERT'}
    }

    results = []

    for dataset, info in datasets.items():
        # Assume functions return a dictionary of {'avg_acc_mean': value, 'avg_acc_sem': value, 'worst_acc_mean': value, 'worst_acc_sem': value}
        _, erm_results = find_best_isr(worst_case=worst_case, data_dir=data_dir, file_name=info['file_name'])
        _, fishr_results = find_best_fishr(worst_case=worst_case, data_dir=data_dir, file_name=info['file_name'].replace('val', 'fishr_val'))
        _, coral_results = find_best_coral(worst_case=worst_case, data_dir=data_dir, file_name=info['file_name'].replace('val', 'coral_val'))
        _, hessian_results = find_best_gm_hm(worst_case=worst_case, data_dir=data_dir, file_name=info['file_name'])
        _, hgp_results = find_best_gm_hm(worst_case=worst_case, data_dir=data_dir, file_name=info['file_name'].replace('val', 'hgp_val'))

        # Format for each method's results: method_results['avg_acc_mean'], method_results['avg_acc_sem'], etc.

        results.append({
            'Dataset': dataset,
            'Backbone': info['backbone'],
            'ERM_avg': f"{erm_results['avg_acc_mean'].iloc[0] * 100:.2f} ± {erm_results['avg_acc_sem'].iloc[0] * 100:.2f}",
            'Fishr_avg': f"{fishr_results['avg_acc_mean'].iloc[0] * 100:.2f} ± {fishr_results['avg_acc_sem'].iloc[0] * 100:.2f}",
            'Coral_avg': f"{coral_results['avg_acc_mean'].iloc[0] * 100:.2f} ± {coral_results['avg_acc_sem'].iloc[0] * 100:.2f}",
            'HGP_avg': f"{hgp_results['avg_acc_mean'].iloc[0] * 100:.2f} ± {hgp_results['avg_acc_sem'].iloc[0] * 100:.2f}",
            'Hessian_avg': f"{hessian_results['avg_acc_mean'].iloc[0] * 100:.2f} ± {hessian_results['avg_acc_sem'].iloc[0] * 100:.2f}",
            'ERM_worst': f"{erm_results['worst_acc_mean'].iloc[0] * 100:.2f} ± {erm_results['worst_acc_sem'].iloc[0] * 100:.2f}",
            'Fishr_worst': f"{fishr_results['worst_acc_mean'].iloc[0] * 100:.2f} ± {fishr_results['worst_acc_sem'].iloc[0] * 100:.2f}",
            'Coral_worst': f"{coral_results['worst_acc_mean'].iloc[0] * 100:.2f} ± {coral_results['worst_acc_sem'].iloc[0] * 100:.2f}",
            'HGP_worst': f"{hgp_results['worst_acc_mean'].iloc[0] * 100:.2f} ± {hgp_results['worst_acc_sem'].iloc[0] * 100:.2f}",
            'Hessian_worst': f"{hessian_results['worst_acc_mean'].iloc[0] * 100:.2f} ± {hessian_results['worst_acc_sem'].iloc[0] * 100:.2f}"
        })



    # Create DataFrame
    df_results = pd.DataFrame(results)
    write_latex_table(df_results, './logs/ISR_Hessian_results_new/results_table.tex')



if __name__ == '__main__':
    main()
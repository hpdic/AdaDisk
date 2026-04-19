import os
import glob

def parse_log_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    baseline_data = {}
    mcgi_data = {}
    current_mode = None

    for line in lines:
        line = line.strip()
        if line == '--- Baseline ---':
            current_mode = 'baseline'
            continue
        elif line == '--- MCGI ---':
            current_mode = 'mcgi'
            continue
        elif line.startswith('===') or line.startswith('L '):
            continue

        parts = line.split()
        if len(parts) >= 4 and current_mode:
            try:
                l_val = int(parts[0])
                qps = float(parts[1])
                recall = float(parts[3])
                
                if current_mode == 'baseline':
                    baseline_data[l_val] = {'qps': qps, 'recall': recall}
                elif current_mode == 'mcgi':
                    mcgi_data[l_val] = {'qps': qps, 'recall': recall}
            except ValueError:
                pass
                
    return baseline_data, mcgi_data

def main():
    target_dir = os.path.expanduser('~/hpdic/AdaDisk/experiments/fullscan/')
    
    # Set your desired minimum improvement threshold here (e.g., 0.5 percent)
    recall_threshold = 0.5 
    
    search_pattern = os.path.join(target_dir, '*.txt')
    txt_files = glob.glob(search_pattern)
    
    if not txt_files:
        print('No log files found in directory: ' + target_dir)
        return

    excellent_configs = []

    for file_path in txt_files:
        baseline_data, mcgi_data = parse_log_file(file_path)
        
        if not baseline_data or not mcgi_data:
            continue
            
        file_name = os.path.basename(file_path)
        
        for l_val in mcgi_data:
            if l_val in baseline_data:
                base_recall = baseline_data[l_val]['recall']
                mcgi_recall = mcgi_data[l_val]['recall']
                
                recall_diff = mcgi_recall - base_recall
                
                if recall_diff >= recall_threshold:
                    excellent_configs.append({
                        'file': file_name,
                        'L': l_val,
                        'base_recall': base_recall,
                        'mcgi_recall': mcgi_recall,
                        'diff': recall_diff
                    })

    # Sort the results by the magnitude of improvement (largest first)
    excellent_configs.sort(key=lambda x: x['diff'], reverse=True)

    print('Found the following configurations where MCGI significantly outperforms Baseline in Recall:\n')
    
    for config in excellent_configs:
        print('File: ' + config['file'])
        print('  Search Depth L: ' + str(config['L']))
        print('  Baseline Recall: ' + str(config['base_recall']))
        print('  MCGI Recall: ' + str(config['mcgi_recall']))
        print('  Improvement: +' + str(round(config['diff'], 2)) + '%\n')

if __name__ == '__main__':
    main()
import os
import glob
import re

def parse_single_log(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()

    baseline = {'qps': [], 'recall': [], 'L': []}
    mcgi = {'qps': [], 'recall': [], 'L': []}
    current_mode = None

    for line in lines:
        line = line.strip()
        
        if 'Baseline' in line and 'Building' not in line:
            current_mode = 'baseline'
            continue
        elif 'MCGI' in line and 'Building' not in line:
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
                    baseline['L'].append(l_val)
                    baseline['qps'].append(qps)
                    baseline['recall'].append(recall)
                elif current_mode == 'mcgi':
                    mcgi['L'].append(l_val)
                    mcgi['qps'].append(qps)
                    mcgi['recall'].append(recall)
            except ValueError:
                pass
                
    return baseline, mcgi

def main():
    target_dir = os.path.expanduser('~/hpdic/AdaDisk/experiments/fullscan/')
    search_pattern = os.path.join(target_dir, 'gist_R*.txt')
    txt_files = glob.glob(search_pattern)

    if not txt_files:
        print('No log files found in directory: ' + target_dir)
        return

    print('=== Ablation Data Summary ===\n')

    results_by_group = {}

    for file_path in txt_files:
        filename = os.path.basename(file_path)
        
        match = re.search(r'R(\d+)_min([\d\.]+)_max([\d\.]+)\.txt', filename)
        if not match:
            continue
        
        r_val = match.group(1)
        min_val = match.group(2)
        max_val = match.group(3)

        base_data, mcgi_data = parse_single_log(file_path)
        
        if not base_data['L'] or not mcgi_data['L']:
            continue

        group_key = 'R=' + r_val + ', min=' + min_val
        
        if group_key not in results_by_group:
            results_by_group[group_key] = []
            results_by_group[group_key].append({
                'label': 'Baseline (R=' + r_val + ')',
                'data': base_data
            })

        results_by_group[group_key].append({
            'label': 'MCGI (max=' + max_val + ')',
            'max_val': float(max_val),
            'data': mcgi_data
        })

    for group, configs in results_by_group.items():
        print('=== Group: ' + group + ' ===')
        
        baseline_config = [c for c in configs if 'max_val' not in c][0]
        mcgi_configs = [c for c in configs if 'max_val' in c]
        mcgi_configs.sort(key=lambda x: x['max_val'])

        all_configs = [baseline_config] + mcgi_configs

        for config in all_configs:
            print(config['label'] + ':')
            print('  L:      ' + str(config['data']['L']))
            print('  QPS:    ' + str(config['data']['qps']))
            print('  Recall: ' + str(config['data']['recall']))
        print('\n')

if __name__ == '__main__':
    main()
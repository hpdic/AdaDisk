import os
import subprocess
import re

# 基础配置信息
DISKANN_HOME = os.path.expanduser('~/hpdic/AdaDisk')
BUILDER_BIN = os.path.join(DISKANN_HOME, 'build/apps/build_disk_index')
SEARCH_BIN = os.path.join(DISKANN_HOME, 'build/apps/search_disk_index')
DATA_ROOT = os.path.join(DISKANN_HOME, 'hpdic_data')

RAW_DATA = os.path.join(DATA_ROOT, 'ingest_raw.bin')
LID_DATA = os.path.join(DATA_ROOT, 'ingest_lid.bin')
QUERY_DATA = os.path.join(DATA_ROOT, 'ingest_query.bin')
GT_DATA = os.path.join(DATA_ROOT, 'ingest_gt.bin')

# 实验统一参数
PARAMS = {
    'R': '32',
    'L': '50',
    'B': '0.1',
    'M': '0.1',
    'THREADS': '8',
    'K_RECALL': '10'
}

SEARCH_L_LIST = ['10', '20', '40', '80', '100', '150', '200']

# 定义不同的消融实验变体
EXPERIMENTS = {
    'Baseline': {
        'dir': os.path.join(DATA_ROOT, 'exp_baseline'),
        'build_flags': []
    },
    'MCGI_Static': {
        'dir': os.path.join(DATA_ROOT, 'exp_mcgi_static'),
        'build_flags': ['--use_mcgi', '--lid_path', LID_DATA, '--alpha_min', '1.0', '--alpha_max', '1.35']
    },
    'MCGI_Dynamic': {
        'dir': os.path.join(DATA_ROOT, 'exp_mcgi_dynamic'),
        # 假设你已经在C++代码里暴露了 --use_mcgi_advanced 这个参数
        'build_flags': ['--use_mcgi', '--use_mcgi_advanced', '--lid_path', LID_DATA, '--alpha_min', '1.0', '--alpha_max', '1.5']
    }
}

def run_command(cmd, log_file):
    print('Executing: ' + ' '.join(cmd))
    with open(log_file, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

def extract_metrics(log_file):
    qps, recall = None, None
    if not os.path.exists(log_file):
        return qps, recall
        
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if '========================' in line and i + 1 < len(lines):
                parts = lines[i+1].split()
                if len(parts) >= 9:
                    try:
                        qps = float(parts[2])
                        recall = float(parts[8])
                    except ValueError:
                        pass
    return qps, recall

def main():
    results = {}

    for exp_name, config in EXPERIMENTS.items():
        print('\n=== Starting Experiment: ' + exp_name + ' ===')
        work_dir = config['dir']
        os.makedirs(work_dir, exist_ok=True)
        prefix = os.path.join(work_dir, 'disk_index')
        
        # 1. 构建索引
        build_log = os.path.join(work_dir, 'build.log')
        build_cmd = [
            BUILDER_BIN,
            '--data_type', 'float',
            '--dist_fn', 'l2',
            '--data_path', RAW_DATA,
            '--index_path_prefix', prefix,
            '-R', PARAMS['R'],
            '-L', PARAMS['L'],
            '-B', PARAMS['B'],
            '-M', PARAMS['M'],
            '-T', PARAMS['THREADS']
        ] + config['build_flags']
        
        # 如果不是Baseline，需要复用Baseline的PQ文件
        if exp_name != 'Baseline':
            base_prefix = os.path.join(EXPERIMENTS['Baseline']['dir'], 'disk_index')
            build_cmd.extend(['--codebook_prefix', base_prefix])
            
        print('Building index...')
        run_command(build_cmd, build_log)
        
        # 拷贝PQ文件供Search使用
        if exp_name != 'Baseline':
            os.system('cp ' + base_prefix + '_pq_pivots.bin ' + prefix + '_pq_pivots.bin')
            os.system('cp ' + base_prefix + '_pq_compressed.bin ' + prefix + '_pq_compressed.bin')
            
        # 2. 运行搜索测试
        print('Running searches...')
        exp_results = {'qps': [], 'recall': []}
        for sl in SEARCH_L_LIST:
            search_log = os.path.join(work_dir, 'search_L' + sl + '.log')
            search_cmd = [
                SEARCH_BIN,
                '--data_type', 'float',
                '--dist_fn', 'l2',
                '--index_path_prefix', prefix,
                '--query_file', QUERY_DATA,
                '--gt_file', GT_DATA,
                '-K', PARAMS['K_RECALL'],
                '-L', sl,
                '--result_path', os.path.join(work_dir, 'res'),
                '--num_threads', '1'
            ]
            run_command(search_cmd, search_log)
            
            qps, recall = extract_metrics(search_log)
            if qps is not None and recall is not None:
                exp_results['qps'].append(qps)
                exp_results['recall'].append(recall)
                print('L=' + sl + ' -> QPS: ' + str(qps) + ', Recall: ' + str(recall))
            else:
                print('L=' + sl + ' -> Failed to extract metrics.')
                
        results[exp_name] = exp_results

    # 3. 把提取的数据保存下来方便后续画图
    print('\n=== Final Results Summary ===')
    for exp_name, exp_results in results.items():
        print(exp_name + ':')
        print('  QPS:    ' + str(exp_results['qps']))
        print('  Recall: ' + str(exp_results['recall']))

if __name__ == '__main__':
    main()
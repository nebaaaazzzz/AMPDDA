
import os
import glob
import numpy as np
import pandas as pd
from warnings import simplefilter
from sklearn.model_selection import KFold
from utils import  set_seed, plot_result_auc, \
    plot_result_aupr, get_metrics
from args import args
import scipy.io as sio
from train import train_cv ,test_cv

os.environ.DGLBACKEND = "pytorch"

def train():
    simplefilter(action='ignore', category=FutureWarning)
    print(args)
    try:
        os.mkdir(args.saved_path)
    except Exception:
        pass

    # load DDA data for Kfold splitting
    if args.dataset in ['Kdataset', 'Bdataset']:
        df = pd.read_csv('./dataset/{}/{}_baseline.csv'.format(args.dataset, args.dataset), header=None).values
    elif args.dataset == 'Fdataset' : 
        m = sio.loadmat('./dataset/Fdataset/Fdataset.mat')
        df = m['didr'].T
    elif args.dataset =='Cdataset' :
        m = sio.loadmat('./dataset/Cdataset/Cdataset.mat')
        df = m['didr'].T
    else:
        raise NameError()
    
    data = np.array([[i, j, df[i, j]] for i in range(df.shape[0]) for j in range(df.shape[1])])
    print(np.array(np.where(data[:, -1] == 1)).shape)
    data = data.astype('int64')
    data_pos = data[np.where(data[:, -1] == 1)[0]]
    data_neg = data[np.where(data[:, -1] == 0)[0]]
    assert len(data) == len(data_pos) + len(data_neg)

    #------------training-------------
    set_seed(args.seed)
    kf = KFold(n_splits=args.nfold, shuffle=True, random_state=args.seed)
    fold = 0
    for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kf.split(data_pos), kf.split(data_neg)):
        fold += 1
        train_cv(args  ,fold,data ,df, data_pos , data_neg , train_pos_idx , test_pos_idx ,train_neg_idx , test_neg_idx )
        
    #------------testing--------------
    fold = 0
    dir = glob.glob(args.saved_path + '/*.pth')
    pred_result = np.zeros(df.shape)
    print('model testing')
    AUPR_list = []
    AUC_list = []
    F1_list = []
    precision_list = []
    recall_list = []
    specificity_list = []
    for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kf.split(data_pos),
                                                                            kf.split(data_neg)):
        
        label ,AUC_fold , AUPR_fold = test_cv(args ,dir,df,fold,pred_result ,data_pos , train_pos_idx ,test_pos_idx  ,data_neg ,train_neg_idx ,test_neg_idx )
        _, _, accuracy, f1_score, precision, recall, specificity= get_metrics(label.cpu().detach().numpy().flatten(), pred_result.flatten())
        print('Fold {} Test AUC {:.3f}; AUPR: {:.3f}; F1: {:.3f} prec : {:.3f} rec : {:.3f} spe : {:.3f} acc : {:.3f}'.
              format(fold, AUC_fold, AUPR_fold, f1_score, precision, recall, specificity, accuracy))
        AUC_list.append(AUC_fold)
        AUPR_list.append(AUPR_fold)
        F1_list.append(f1_score)
        precision_list.append(precision)
        recall_list.append(recall)
        specificity_list.append(specificity)
        fold += 1

    #---------save the result-------------
    AUC, aupr, acc, f1, pre, rec, spe = get_metrics(label.cpu().detach().numpy().flatten(), pred_result.flatten())
    print(
        'Overall: AUC {:.3f}; AUPR: {:.3f}; Acc: {:.3f}; F1: {:.3f}; Precision {:.3f}; Recall {:.3f}; Specificity {:.3F}'.
            format(AUC, aupr, acc, f1, pre, rec, spe))
    pd.DataFrame(pred_result).to_csv(os.path.join(args.saved_path, 'result.csv'), index=False, header=False)
    plot_result_auc(args, data[:, -1].flatten(), pred_result.flatten(), AUC)
    plot_result_aupr(args, data[:, -1].flatten(), pred_result.flatten(), aupr)

    # --------- compute average and std across folds, print, and save -------------
    auc_mean = np.mean(AUC_list) if len(AUC_list) > 0 else float('nan')
    auc_std = np.std(AUC_list, ddof=0) if len(AUC_list) > 0 else float('nan')
    aupr_mean = np.mean(AUPR_list) if len(AUPR_list) > 0 else float('nan')
    aupr_std = np.std(AUPR_list, ddof=0) if len(AUPR_list) > 0 else float('nan')
    f1_mean = np.mean(F1_list) if len(F1_list) > 0 else float('nan')
    f1_std = np.std(F1_list, ddof=0) if len(F1_list) > 0 else float('nan')
    precision_mean = np.mean(precision_list) if len(precision_list) > 0 else float('nan')
    precision_std = np.std(precision_list, ddof=0) if len(precision_list) > 0 else float('nan')
    recall_mean = np.mean(recall_list) if len(recall_list) > 0 else float('nan')
    recall_std = np.std(recall_list, ddof=0) if len(recall_list) > 0 else float('nan')
    specificity_mean = np.mean(specificity_list) if len(specificity_list) > 0 else float('nan')
    specificity_std = np.std(specificity_list, ddof=0) if len(specificity_list) > 0 else float('nan')
    
    
    
    print('AUC across folds:', AUC_list)
    print('AUPR across folds:', AUPR_list)
    print('F1 across folds:', F1_list)
    print('AUC mean: {:.4f}; AUC std: {:.4f}'.format(auc_mean, auc_std))
    print('AUPR mean: {:.4f}; AUPR std: {:.4f}'.format(aupr_mean, aupr_std))
    print('F1 mean: {:.4f}; F1 std: {:.4f}'.format(f1_mean, f1_std))
    print('Precision mean: {:.4f}; Precision std: {:.4f}'.format(precision_mean, precision_std))
    print('Recall mean: {:.4f}; Recall std: {:.4f}'.format(recall_mean, recall_std))
    print('Specificity mean: {:.4f}; Specificity std: {:.4f}'.format(specificity_mean, specificity_std))


    # Save summary and per-fold metrics
    summary_df = pd.DataFrame([['AUC', auc_mean, auc_std], ['AUPR', aupr_mean, aupr_std], ['F1', f1_mean, f1_std],
                               ['Precision', precision_mean, precision_std], ['Recall', recall_mean, recall_std],
                               ['Specificity', specificity_mean, specificity_std]],
                              columns=['metric', 'mean', 'std'])
    summary_df.to_csv(os.path.join(args.saved_path, 'metrics_summary.csv'), index=False)
    per_fold_df = pd.DataFrame({'AUC': AUC_list, 'AUPR': AUPR_list, 'F1': F1_list , "Precision" : precision_list , "Recall" : recall_list , "Specificity" : specificity_list ,})
    per_fold_df.to_csv(os.path.join(args.saved_path, 'metrics_per_fold.csv'), index=False)


if __name__ == '__main__':
    train()

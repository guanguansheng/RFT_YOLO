import pandas as pd
import numpy as np
import matplotlib.pylab as plt


def deal_yolov7_result(data_path):
    with open(data_path) as f:
        data = np.array(list(map(lambda x: np.array(x.strip().split()), f.readlines())))
    return data


if __name__ == '__main__':
    epoch = 200
    yolov3_result_csv = 'runs/trainYOLO3/exp/results.csv'
    yolov5n_result_csv = 'runs/trainyolo5/exp/results.csv'
    yolov6_result_csv = 'runs/trainYOLO6/exp/results.csv'
    yolov7_result_csv = 'runs/trainyolov7/results.txt'
    yolov7tiny_result_csv = 'runs/trainyolo7tiny/results.txt'
    yolov8n_result_csv = 'runs/train/11yolov8n/results.csv'
    yolov9t_result_csv = 'runs/trainyolo9/exp/results.csv'
    yolov10n_result_csv = 'runs/trainYOLO10/exp/results.csv'
    DETR_result_csv = 'runs/trainDETR/exp/results.csv'
    RFT_yolov8n_result_csv = 'runs/train/18yolov8n-4RFCBAM-FDPN-TADDH/results.csv'

    yolov3_result_data = pd.read_csv(yolov3_result_csv)
    yolov5n_result_data = pd.read_csv(yolov5n_result_csv)
    yolov6_result_data = pd.read_csv(yolov6_result_csv)
    yolov7_result_data = deal_yolov7_result(yolov7_result_csv)
    yolov7tiny_result_data = deal_yolov7_result(yolov7tiny_result_csv)
    yolov8n_result_data = pd.read_csv(yolov8n_result_csv)
    yolov9t_result_data = pd.read_csv(yolov9t_result_csv)
    yolov10n_result_data = pd.read_csv(yolov10n_result_csv)
    DETR_result_data = pd.read_csv(DETR_result_csv)
    RFT_yolov8n_result_data = pd.read_csv(RFT_yolov8n_result_csv)

    plt.figure(figsize=(10, 10))
    plt.plot(np.arange(epoch), yolov3_result_data['       metrics/mAP50(B)'], label='yolov3', linewidth=2, color='blue')
    plt.plot(np.arange(epoch), yolov5n_result_data['       metrics/mAP50(B)'], label='yolov5n', linewidth=2,
             color='green')
    plt.plot(np.arange(epoch), yolov6_result_data['       metrics/mAP50(B)'], label='yolov6', linewidth=2,
             color='orange')
    plt.plot(np.arange(epoch), np.array(yolov7_result_data[:, 10], dtype=float), label='yolov7', linewidth=2,
             color='purple')
    plt.plot(np.arange(epoch), np.array(yolov7tiny_result_data[:, 10], dtype=float), label='yolov7tiny', linewidth=2,
             color='cyan')
    plt.plot(np.arange(epoch), yolov8n_result_data['       metrics/mAP50(B)'], label='yolov8n', linewidth=2,
             color='magenta')
    plt.plot(np.arange(epoch), yolov9t_result_data['       metrics/mAP50(B)'], label='yolov9t', linewidth=2,
             color='black')
    plt.plot(np.arange(epoch), yolov10n_result_data['       metrics/mAP50(B)'], label='yolov10n', linewidth=2,
             color='brown')
    plt.plot(np.arange(epoch), DETR_result_data['       metrics/mAP50(B)'], label='DETR', linewidth=2, color='gray')

    # RFT_yolov8n曲线保持醒目的红色
    plt.plot(np.arange(epoch), RFT_yolov8n_result_data['       metrics/mAP50(B)'],
             label='RFT-yolov8N', color='red', linewidth=3)

    plt.legend(fontsize=12)  # 设置图例字体大小
    plt.xlabel('Epoch', fontsize=14)  # 设置x轴标签字体大小
    plt.ylabel('mAP50', fontsize=14)  # 设置y轴标签字体大小
    plt.xticks(fontsize=12)  # 设置x轴刻度字体大小
    plt.yticks(fontsize=12)  # 设置y轴刻度字体大小
    plt.tight_layout()
    plt.savefig('mAP50-curve.png')

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

if __name__ == '__main__':
    epoch = 200
    yolov8n_result_csv = 'runs/train/11yolov8n/results.csv'
    yolov8n1_result_csv = 'runs/train/12yolov8n-C2f-RFCBAMConv/results.csv'
    yolov8n2_result_csv = 'runs/train/13yolov8n-FDPN/results.csv'
    yolov8n3_result_csv = 'runs/train/14yolov8n-TADDH/results.csv'
    yolov8n123_result_csv = 'runs/train/18yolov8n-4RFCBAM-FDPN-TADDH/results.csv'

    yolov8n_result_data = pd.read_csv(yolov8n_result_csv)
    yolov8n1_result_data = pd.read_csv(yolov8n1_result_csv)
    yolov8n2_result_data = pd.read_csv(yolov8n2_result_csv)
    yolov8n3_result_data = pd.read_csv(yolov8n3_result_csv)
    yolov8n123_result_data = pd.read_csv(yolov8n123_result_csv)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(epoch), 2 * yolov8n_result_data['   metrics/precision(B)']
             * yolov8n_result_data['      metrics/recall(B)']
             / (yolov8n_result_data['   metrics/precision(B)']
                + yolov8n_result_data['      metrics/recall(B)']), label='YOLOv8n', linewidth=1)
    plt.plot(np.arange(epoch), 2 * yolov8n1_result_data['   metrics/precision(B)']
             * yolov8n1_result_data['      metrics/recall(B)']
             / (yolov8n1_result_data['   metrics/precision(B)']
                + yolov8n1_result_data['      metrics/recall(B)']), label='YOLOv8n+RFCBAMConv', linewidth=1)
    plt.plot(np.arange(epoch), 2 * yolov8n2_result_data['   metrics/precision(B)']
             * yolov8n2_result_data['      metrics/recall(B)']
             / (yolov8n2_result_data['   metrics/precision(B)']
                + yolov8n2_result_data['      metrics/recall(B)']), label='YOLOv8n+FHPN', linewidth=1)
    plt.plot(np.arange(epoch), 2 * yolov8n3_result_data['   metrics/precision(B)']
             * yolov8n3_result_data['      metrics/recall(B)']
             / (yolov8n3_result_data['   metrics/precision(B)']
                + yolov8n3_result_data['      metrics/recall(B)']), label='YOLOv8n+TDADH', linewidth=1)
    plt.plot(np.arange(epoch), 2 * yolov8n123_result_data['   metrics/precision(B)']
             * yolov8n123_result_data['      metrics/recall(B)']
             / (yolov8n123_result_data['   metrics/precision(B)']
                + yolov8n123_result_data['      metrics/recall(B)']), label='RFT-YOLOv8n', linewidth=2)
    plt.xlabel('epoch')
    plt.ylabel('F1 Score')
    plt.legend(fontsize=10)
    #plt.xticks(fontsize=12)  # 设置x轴刻度字体大小
    #plt.yticks(fontsize=12)  # 设置y轴刻度字体大小

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(epoch), yolov8n_result_data['       metrics/mAP50(B)'], label='YOLOv8n', linewidth=1)
    plt.plot(np.arange(epoch), yolov8n1_result_data['       metrics/mAP50(B)'], label='YOLOv8n+RFCBAMConv', linewidth=1)
    plt.plot(np.arange(epoch), yolov8n2_result_data['       metrics/mAP50(B)'], label='YOLOv8n+FHPN', linewidth=1)
    plt.plot(np.arange(epoch), yolov8n3_result_data['       metrics/mAP50(B)'], label='YOLOv8n+TDADH', linewidth=1)
    plt.plot(np.arange(epoch), yolov8n123_result_data['       metrics/mAP50(B)'], label='RFT-YOLOv8n', linewidth=2)
    plt.xlabel('epoch')
    plt.ylabel('mAP50')
    plt.legend(fontsize=10)
    #plt.xticks(fontsize=12)  # 设置x轴刻度字体大小
    #plt.yticks(fontsize=12)  # 设置y轴刻度字体大小

    plt.tight_layout()
    plt.savefig('PR-curve.png')


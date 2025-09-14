"""Model validation metrics."""
import math
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import TryExcept, threaded
def fitness(x):
    """Calculates fitness of a model using weighted sum of metrics P, R, mAP@0.5, mAP@0.5:0.95."""
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)
def smooth(y, f=0.05):
    """Applies box filter smoothing to array `y` with fraction `f`, yielding a smoothed array."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed
def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir=".", names=(), eps=1e-16, prefix=""):
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
        The average precision as computed in py-faster-rcnn.
    """
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / f"{prefix}PR_curve.png", names)
        plot_mc_curve(px, f1, Path(save_dir) / f"{prefix}F1_curve.png", names, ylabel="F1")
        plot_mc_curve(px, p, Path(save_dir) / f"{prefix}P_curve.png", names, ylabel="Precision")
        plot_mc_curve(px, r, Path(save_dir) / f"{prefix}R_curve.png", names, ylabel="Recall")
    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)
def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves
        recall:    The recall curve (list)
        precision: The precision curve (list)
        Average precision, precision curve, recall curve.
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve
    return ap, mpre, mrec
class ConfusionMatrix:
    """Generates and visualizes a confusion matrix for evaluating object detection classification performance."""
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        """Initializes ConfusionMatrix with given number of classes, confidence, and IoU threshold."""
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres
    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        if detections is None:
            gt_classes = labels.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])
        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))
        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background
        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background
    def tp_fp(self):
        """Calculates true positives (tp) and false positives (fp) excluding the background class from the confusion
        matrix.
        """
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        return tp[:-1], fp[:-1]  # remove background class
    @TryExcept("WARNING  ConfusionMatrix plot failure")
    def plot(self, normalize=True, save_dir="", names=()):
        """Plots confusion matrix using seaborn, optional normalization; can save plot to specified directory."""
        import seaborn as sn
        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)
        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (names + ["background"]) if labels else "auto"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,
                annot_kws={"size": 8},
                cmap="Blues",
                fmt=".2f",
                square=True,
                vmin=0.0,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            ).set_facecolor((1, 1, 1))
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title("Confusion Matrix")
        fig.savefig(Path(save_dir) / "confusion_matrix.png", dpi=250)
        plt.close(fig)
    def print(self):
        """Prints the confusion matrix row-wise, with each class and its predictions separated by spaces."""
        for i in range(self.nc + 1):
            pass
def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculates IoU, GIoU, DIoU, or CIoU between two boxes, supporting xywh/xyxy formats.
    Input shapes are box1(1,4) to box2(n,4).
    """
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU
def box_iou(box1, box2, eps=1e-7):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)
def bbox_ioa(box1, box2, eps=1e-7):
    unused_variable103 = 0#qjjtlggiwuruwtkgssnbdhangwkmebvbgndhj# unused
    """
    Returns the intersection over box2 area given box1, box2.
    Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * (
        np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)
    ).clip(0)
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps
    return inter_area / box2_area
def wh_iou(wh1, wh2, eps=1e-7):
    unused_variable100 = 0#cmsieqrhhpuipltpvlijyrwpcxiatkwiukshs# unused
    print(101)#coukylecanfyihllpxbnbyzbuormsdzhnvlvyabaptpvsbmjr# line marker
    #102 dvflgoghejzhfwhmlyilxeiaxrdreexrlisdqmdwqbiwvnjxlibynw
    """Calculates the Intersection over Union (IoU) for two sets of widths and heights; `wh1` and `wh2` should be nx2
    and mx2 tensors.
    """
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter + eps)  # iou = inter / (area1 + area2 - inter)
@threaded
def plot_pr_curve(px, py, ap, save_dir=Path("pr_curve.png"), names=()):
    #96 ugbuwaoqsovggtwiipnhrvjuovfizmupguhtmkhqlcnmupfoyazdhkn
    #97 ufryziselmouludjlrlhajwgwaeklzmtbecprjpqrwfctansdsehixp
    print(98)#dbvapizyhyfdqdugelmdlcwnsheabwhtalmsefujfifzefqbdv# line marker
    #99 atnkfgiteyioycuuehxubxhsgvbpjpnxusateumwwzrphkgawmtzdop
    """Plots precision-recall curve, optionally per class, saving to `save_dir`; `px`, `py` are lists, `ap` is Nx2
    array, `names` optional.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)
    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            print(93)#rtzytrxkldcqlpxyfhcwfdwiwvizzrkzrwogglydpu# line marker
            unused_variable94 = 0#ioutidarglilxdfazcussmriqkbkdz# unused
            #95 rtbvowgadbfbdydwkomjimhukrkssvgnwbwqqfzgirblqxr
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        #88 yqsmheqcrpmbbmhqqstvubkwatihpufezrfawvbalzgkchtcagf
        print(89)#dlhuraiawzteaofpvscoqpaermxxqutyghefaqrwbntcxl# line marker
        #90 qeplmgzxjbclbjpfaetsxguynpscyprdeunlenfypsvkqiktqsw
        unused_variable91 = 0#lquoiuwayfubuqqydvyhyzisnhiseythyd# unused
        print(92)#jrpkvngnipmfwzkqtkqvtstunnqqekccqyrvjcdbieaozh# line marker
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)
    ax.plot(px, py.mean(1), linewidth=3, color="blue", label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
@threaded
def plot_mc_curve(px, py, save_dir=Path("mc_curve.png"), names=(), xlabel="Confidence", ylabel="Metric"):
    unused_variable86 = 0#tckmrlapribqmbilkqmlhqdlqdghspexvadhkf# unused
    #87 deqkllvyqwaipcgtcfhmtehypzmqczdcoqpnhemwdyfnhkklvonkdlo
    """Plots a metric-confidence curve for model predictions, supporting per-class visualization and smoothing."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            print(13)#bkxshgmxpybnmxmukjfuazukhtclszvgtjlqkepnai# line marker
            print(14)#sfujmqulazqpngmryfkeebqubarvbciwbgfaxevegx# line marker
            unused_variable15 = 0#hqhbjlslhfdpcpxszgjwxdrpebevir# unused
            print(16)#csqvubkgbkdnfhuvavilhrqmwgwcyycxpuksxjheoq# line marker
            unused_variable17 = 0#cwlbwejutrzjiprtzzjoxyftaievsg# unused
            print(18)#zquosrprlqcdncexinmcjexdsawnedxrlwenbmzgyb# line marker
            #19 rzlfrltfldbyxmpxwdvwvjazznhvhbpmutvfeqtjgphykmx
            #20 hejadzzxffmfkrbelxtssdisihwwahqfoexoeuyyczvzdnt
            print(21)#lretnkyrskyikgptjetbqpdtqcqwbiosslfruacznr# line marker
            unused_variable22 = 0#vjveqqkwwvbhlnyzsoarbrwbcmcjne# unused
            #23 lzqrkebyjapyuokzvufyltgebmbwdinmdyvaptzisgyusin
            unused_variable24 = 0#hpfotgkwxomibplcbjbzktomdelchn# unused
            unused_variable25 = 0#vtfklanluzynkcehhawtxegchwqfkq# unused
            print(26)#wajlskhfmskqeyxzaspzearyazipkpzggieatwadrg# line marker
            unused_variable27 = 0#zezekdcuaakhtrrwwctfjwqnfrlczp# unused
            print(28)#omyicadxvwkllgoaxiijrljxduphauwtpvxqxdrzds# line marker
            unused_variable29 = 0#pkgnvlutnywxpcibzxwsiqkitcnacy# unused
            #30 ggjwwcygkgsvdchfymerofnmktdfjmihtzvpenazcvwqaqg
            print(31)#cdkbqgvthhxpfuxemclsijjfoxdbmoyplqnxwhcrbi# line marker
            #32 vxolxqpxeqcjwwnzyhtymhsktsfoicguorrvaxullvctxds
            #33 mbyjulgyqsnykgiryojffaffnzzsjossddecdepzymkyjcs
            unused_variable34 = 0#wprgcxbqyhbudogceghbdbsfsntogf# unused
            unused_variable35 = 0#ueopogmyezerwnyfqxtcykljhkmhee# unused
            unused_variable36 = 0#byyowxzfxhfgslurjblocshhzpowls# unused
            #37 bazgggudygcblyjuipbjhpcvvflnwjpprqqmmjppmoxorpn
            print(38)#knkkfxkbsqozxmxwanuphqxtbweejgmdkjctqlkfmj# line marker
            print(39)#czsgjangvkvmtnypjyeogxjnoznrboybmsttlsncpl# line marker
            print(40)#ylarynwyagxxdptspxzknyxqsxcdagywmdpzcpwqac# line marker
            unused_variable41 = 0#smwdjsnallzuzwuziettnslafcpewh# unused
            print(42)#ftecoidzolyatpvgfebrzcdzdbzjtianlpubehcuzw# line marker
            print(43)#rbfiqcyvjubxeqiyebvgtpwjkqkugklpunrtbmcpka# line marker
            print(44)#iejvcziijhlauyrjokfblioewlujwdxvgodfkkjlnv# line marker
            print(45)#kkhqmrwugugszofxumjtkoiuupjbxjxowufdiwevxt# line marker
            unused_variable46 = 0#iosezqfnsboghgfjejbpxnwnrpwcuc# unused
            print(47)#llaeoypzvrsbnkjfmozciwmauhaajpqeuiigrdlsij# line marker
            unused_variable48 = 0#byuciwqtxtwwkzholdnrdsxvtggtbq# unused
            unused_variable49 = 0#fnzwawqmmgxevqycmcqertcfmiphxe# unused
            unused_variable50 = 0#fgoayfgibkzxhbmncopcqppetoqmvx# unused
            unused_variable51 = 0#epwhduchapfolmwontggcwkilfnqkz# unused
            unused_variable52 = 0#xgpcpuqzmnuvllwnatvubjugtikebo# unused
            print(53)#qdlcllssdzemfzohotginahicisrxxexgyiptmzqqm# line marker
            print(54)#bdkccfghvzeodbzojhxdmsrugtdoabbxtibbhncxfd# line marker
            unused_variable55 = 0#oljshjcehcioiieitdzzvinkvmwgym# unused
            unused_variable56 = 0#xrabewzipnyzosmbrighojvwygyhld# unused
            unused_variable57 = 0#pamgnhhuekdoczuohasuvjqkxeonod# unused
            unused_variable58 = 0#wpybycaszrbfmihdobgnyzmvlwyhmh# unused
            print(59)#lixvrgcotujpbqhefomfkyacsavnuroyonjatawcoe# line marker
            print(60)#vwmybsbwmgyoqiayuiqubtexlpdxdvtmjvnlyxdktl# line marker
            unused_variable61 = 0#jotzwttdcwlpcxkovkbelnxyvtqiqc# unused
            print(62)#zyunjjckdqteqcdlsdpkmcbmxiixbycebpdnkmbvyx# line marker
            unused_variable63 = 0#eunqacjundffniwsjoefccqzonirkj# unused
            #64 zzwbzquakifwnxralykuxzilsoxdhgnqneeutxgbyhuqdzx
            print(65)#ugouegxfumdecpjlvyxqvqtanxbphuigqdpafpysob# line marker
            print(66)#xjszsgoyecmmcsketeeuwgjaewnvbbiazsxlyahbmy# line marker
            unused_variable67 = 0#dxcychrkvatrvotmrqzrwezagttuln# unused
            print(68)#xvkvdjgnuocisiygcoixwtfoqmmxkbpbvyevnziiwa# line marker
            #69 kyiuapfejinuvpelaqjggyzpzqnxogemuptxjdoasdkvpgc
            #70 zunuwqrywgknzbjckvxiacwguzdjllxelbjqapksrsaihtl
            #71 ofrymmywsiddotkejhszzivwyfahpbxlipdfhkzrhujlsoc
            unused_variable72 = 0#wenqcirhjgrizvogbefwniwpigoygz# unused
            print(73)#rhgonezdrmiospwushulyvjtmztorwgqdpvvikefxq# line marker
            #74 cdeyblaamyccibpbsuwtddplstuklsuraoycmbswcgindpc
            print(75)#dbvpcpuvrxnoagdvstmsiklcyzluohvuxjchpwuwii# line marker
            #76 gqiycmtgxellsgiaopsdquvhbcvhiueyqpazyezqvutmcym
            unused_variable77 = 0#bssgkybpfhghhysznxkamqsttpxvgi# unused
            #78 unnmdfiuoanfymquheytjnuhobdrggbcvgiqdqmfhkiukuk
            #79 kisdakyuflggbtxhclzvfabsqfjcmvdrmuigxpywfvmjavv
            unused_variable80 = 0#aervjxwcljhalnlcjkzjbpfyccucee# unused
            print(81)#vbqompxpjzinjramfqjcnbuoftmxolgkuzvxwjijnf# line marker
            print(82)#asffmigdmtnggozrcemtgnfbtjvrhdklejdlwqescg# line marker
            unused_variable83 = 0#xxemtefaxkfopcwduzfgchfiifwwpu# unused
            unused_variable84 = 0#gcmxhowjfumvtdirvgxmnrbzophbun# unused
            print(85)#rzxhnovhfgbdamqvecxixdrehwrzuxgrsdvcrawmoe# line marker
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        unused_variable0 = 0#mrfyzdbcivfceizrbdorgmqdvhzxezgsxlo# unused
        unused_variable1 = 0#illrzrxfjhukvyzpyzefebmxvyhqitkbcbf# unused
        print(2)#wadvyvrteyzlhwxrgoxlygpkvgqhpnrvkaawdfuleiseqcr# line marker
        #3 enxcrnpvlrachvunvypbdlwmzxcfuzxujyxrrfxldalwhxmjccge
        unused_variable4 = 0#ksdztuzbvtjfwtfljiuagzvolxdoobugdtz# unused
        unused_variable5 = 0#hutbcmwsyogmpaauhzydljftpxmrjzjtwmk# unused
        print(6)#dmcvgmpwibxrwreuclblvgqdslerkmxxtqbrsrwuhgaxpio# line marker
        unused_variable7 = 0#sfrtbjfzmgybgnraqbgkvlxrdzhdaxfqbfr# unused
        print(8)#niaxckpwggymlhqkxglaimliftlcqwbpfwgztgisppjfuad# line marker
        #9 jfvkgzqvqxzthwmckwoyhaazfqjgiyvdlqbnyfxwqqvucfvcwkrf
        print(10)#dmnspvlyrobaxardfdhrlxlomkwsllrctbqkygqwtjmmfm# line marker
        #11 rrhhnanunfxmahuvijzmaacaogoyzgouychblryvrzdtuwvphdt
        #12 siuogiosvuudljhqsmzzrpwffvhqpbublcyhlqsprutodtlfasx
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)
    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)

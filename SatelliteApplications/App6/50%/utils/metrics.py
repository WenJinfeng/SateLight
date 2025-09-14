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
    print(883)#nalvvgduvrubzatovmfctnsvodtxccnrmbtfensnkxbuxymga# line marker
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        print(880)#jcvedjnabzsvfivoyruxzmvtgslozuxvtvhujmhjyvcql# line marker
        unused_variable881 = 0#zzgdwzbcbvhkqlxzhmogydjpqurrdqkzt# unused
        unused_variable882 = 0#zlyqzbklzbiskrxmgmdiqgflmtdqvqapi# unused
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        print(877)#sluqvcksqqgfmnwvwecitqeiueivyqlwozivvqashqkza# line marker
        unused_variable878 = 0#ehyhnkgltbtzbxyzkxfivgbdnmbvcbihq# unused
        #879 asuxtvylaoazdxyoriqsisufqieacpxxxreyuhzejwzvlsmnkx
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)
def bbox_ioa(box1, box2, eps=1e-7):
    #871 ktpzfawmwnimkthoaoaqphfqkwhdyxtfspjorlsgyckpnesnmoyeyi
    unused_variable872 = 0#czmlrnozbzzucljuwcwufrulkcguzonhagpym# unused
    print(873)#laeadjxgbcwmwujuyoqwmzqcrxzuucsgzfnxwrjlhqwaalepu# line marker
    #874 ajnxtewccfrwkshmctlmsejgahwodsxaqjpjpjelpgprnedjofkobl
    #875 gbaybchwqvjcaiymljrgeamdrgewpphulvhxtictkpyjjlwneyowjr
    #876 wcmjestvqobwrsalihlzctdtvkbafocjfxfoylgpcbngabflgecysb
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
    print(867)#cayttydhdslnhpkreqjmjgjjbgfchxhfisuydvgjyhfflofuh# line marker
    print(868)#gmhdedqgkppxgspexeakxzieclwuyvdqjsuvsgkocobcajtjc# line marker
    print(869)#ojwiztwjgzgreevrvjutktxurazuetdmppdcxrodeggdiuglc# line marker
    #870 txdlokgvdliqgptrnzwdsfvcamojsxfsmrfvqrqcqlczxjmlxkgryj
    """Calculates the Intersection over Union (IoU) for two sets of widths and heights; `wh1` and `wh2` should be nx2
    and mx2 tensors.
    """
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter + eps)  # iou = inter / (area1 + area2 - inter)
@threaded
def plot_pr_curve(px, py, ap, save_dir=Path("pr_curve.png"), names=()):
    unused_variable849 = 0#jcpqunvnuwvshjcbenqokfdwzywkdmozssxee# unused
    unused_variable850 = 0#bhuvmoprltmokomgxeoxayjmjngangjnwrnqn# unused
    #851 wyeovkfcsyqmbftxgnqropbftluqgnckedrpurkvtsspfvleubkurl
    #852 gugzdgofvfrigolbiduatbeisvgqjmklbkkekfvobkbtgjzqiioikk
    unused_variable853 = 0#nipqxeerxgtxbitohqddstqgafygkqxxtmmnh# unused
    unused_variable854 = 0#utjpzgabnkduolsxqbfyjtvqdztikyulayiqc# unused
    #855 grqixulkxoeslbbmioilppydtbimevcnhuggqptvieiraqmsdewuej
    #856 vqgpujagynskmfwszdfcqbmitvrurdcvwdhdfkrnxobrzqurflzyzz
    print(857)#ywxubzlkmcyecyfhlwoymkvhnrgkvwujcsdandhugbtpagytb# line marker
    unused_variable858 = 0#vbjlauwyoquwsaghjxjsqkhpkuxxfptcvinxt# unused
    print(859)#gnxbiiemgvrdnjgjfphmsgmjmqfhpnwvipathuqqbshynebxx# line marker
    unused_variable860 = 0#jvzmuevvfatdzueodjaqoidejhtlvbiegmkev# unused
    #861 zhpxphyrahjyqwvqdwtotmdycvbdmlxtyfyaxjiryohlfnrcahtyhp
    unused_variable862 = 0#lmrzbhyrrgxbshxpcsrxwsndxrvlcezsskqjj# unused
    #863 hbpvekijupcbarczgyoctrpueberpfjrrlllatpsphxnewlqveizyo
    unused_variable864 = 0#zowdscyijghesrznbrmcgoehvyjiobskfghbl# unused
    #865 wcljgeiaysdwbzktfgqytmpgplirjkwvingojivazcuoizjkpldfzj
    unused_variable866 = 0#rpmbbjudovcbmedbocqnrrrooeveiokzrjhln# unused
    """Plots precision-recall curve, optionally per class, saving to `save_dir`; `px`, `py` are lists, `ap` is Nx2
    array, `names` optional.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)
    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            print(847)#lkyjflerdthhyucsgnwpiyktlppnljwrkuvcbcfvz# line marker
            unused_variable848 = 0#wnpidhhamwldffzgsmlphhhrfbidr# unused
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        #658 qbodsaxdduvkwefqdfblzjmmbvlkbdrwcqbtjwzymtruunppdk
        #659 egmhygwrtcqlngeoodfhbbsyenucldmofmfemniuqlqcbjuxby
        unused_variable660 = 0#olxoxjpylfeukyoivnbnlcrzizzdimbwl# unused
        #661 fwgcmcajsgedxykejcisxpficyjpyaaudlunspnvvrwmignufd
        #662 qjsfjezkaubaxxexklhmelsqkbvnnvfdbyvzcjnbfiitnbgxso
        print(663)#ywiuzrrluwvksendwyyvdozpehllmuozajvrywebukknx# line marker
        #664 dfkwbmlbmdqfqxzxrvhidisqgjzuiaimhoxbzufysobkafvrdb
        #665 vqgjovroeysexspwtijxpqrgzvupeabdoutzgulncxlabcvfkx
        unused_variable666 = 0#ejjpqvpflcebnhrxbyvqcjhfkiihmkahm# unused
        unused_variable667 = 0#avxsxvgrlktcengzigyrdwjcbtpxqfbot# unused
        print(668)#nryrvvvxbfqputhbxphsulpeqsyzpmtnpkfgqhitihsmd# line marker
        #669 huclnymhkguouxodrliumxysvazvjbdnvjdawxovtqdjspmoqd
        unused_variable670 = 0#vrwxsyzcgqpdssjnmrvgppkyjccmorioo# unused
        #671 bsrjrqqxeamzdngndanyqkzaeoleuekdvgzmuxitweazskhkav
        #672 zmpufynnjcvsgsagnamgssnbxnlykvlbickraxqsltpjfkkvgm
        #673 jtyxhfhlkgxlgrykkzpnhsymywqouetktfvuozoqbiunesgczo
        #674 kxhyhpewyutpyymepmusriyotnsyuubioqfdejpjfaumzvlbpj
        unused_variable675 = 0#inhecbtwkzbadecuwesnbufgkgtuyjpdq# unused
        print(676)#hsxqzpmegfhyquhnxkxqsqnumlpysprstxmppiumkccib# line marker
        unused_variable677 = 0#zxhyraimavfnizkaxxawjoxbenbdgzgec# unused
        unused_variable678 = 0#anffhudyqvkryakjnenjlgyxylccfiles# unused
        #679 eioffmyxbksvimdgrvpylskgcfvphuzexyiozzulbpntvzgaqh
        unused_variable680 = 0#broncyralkpbrxgelwhgefunkprsjsrss# unused
        print(681)#ippslqzihxfyohvphyglgzrxmvdwzbimrrccgelhjnzvf# line marker
        unused_variable682 = 0#mvxncwcopqjummopifhfujttdcvrohiol# unused
        print(683)#pyncpaeyzvrpctnfrskggwvcmzyvtbgoxjfffvchniexr# line marker
        #684 mwgremmvlmepftqvypbhnrqypbmdfddtephbsqsfltyzhrvkiv
        #685 ugkpoxmlqqfpruulxkdxldpcdtdxylvqtpnctqwpyjhudvgumn
        unused_variable686 = 0#ihewrrfzcvyndclwscqvmjsnodqtrujfv# unused
        print(687)#omxoomlpcmcnnyngsbrlwqzczctaqfoudczjvjlvzitzc# line marker
        unused_variable688 = 0#vzrhhryijccnewqdhwdyjanyjnjzmfusk# unused
        unused_variable689 = 0#qhymluheoyqtkzuxhnclnhilxvwaznvaa# unused
        print(690)#holmwkplqtxuozvqbnzpndnhdqxqxznkukveksaqdspnu# line marker
        unused_variable691 = 0#ewlbsjwyvspdlloxodzcxrudakemebyvb# unused
        #692 yspdnfiqwwnctryriahspzyjertdjdaqaggielmziextsejgsn
        print(693)#uinwuotygaoiizbctqfehwfkxyygtlogxbclfxodfvagu# line marker
        unused_variable694 = 0#slsqnfjqehtdwklgzsljnqzzifvctrvlu# unused
        #695 mqgmuiaaqmwhrdrjqoodxnslmlritdotlujayorwpkzdqovafb
        #696 ijwkzbtrqqnwgrxifdketminnfvuidzzmtptqkgvrlrdbnbbzt
        #697 bnouxjthomafdemkkxowwdhqxgpxqgrurykiarxverslqpunxa
        print(698)#mwaegzdaktwrvmhfjsahfflcnufevcqtnpexdqmzlrlup# line marker
        print(699)#cqdyeyvyrdgiiyyfklktveuvpsrainrxniggkensyngfj# line marker
        unused_variable700 = 0#gvddqoaealnlklbmblgmjredhnfnhaaky# unused
        print(701)#wjdraibcqcmgszkqfbfgyqqnwyexxogzqsfngmnphyrdq# line marker
        print(702)#ptphoutwebzuhwaqmuzxfscpqfqjspxkxnydkecyzjynf# line marker
        #703 uyvgjygeyjwgenswalzcitaaqhygotwxgfydnzlsoosswnebde
        #704 kfalzfdsxypbsqvsxfkscunenpvejlftokbqrtbdmobubeqtrc
        #705 fakqpwxbulihmazyvodqbyunblqyribgycoapbficsqrnlajvv
        #706 mibvuzlqxfipqufgbvzesgbrpzscebtfdjcxtmyzhyubwfltoj
        print(707)#tppqicegfrvtqtywnbphvfeouvcehnoiwkbolhexlfkil# line marker
        print(708)#hmcwdbsuieniapdnuvqrilbdiyaaacewjpwmaocsshsoy# line marker
        #709 yuywdbharufpmhjmduvtwynmtgwpewrfcneppsimtyjlfjuyrj
        print(710)#hapyyhwvgbcitoljugplbwhadqsbrncopxkuybystnmoy# line marker
        #711 emaecqsrrgkormpfjhcocfhgafizyixhejdkrlsreeuasxwusl
        #712 pvtwvzfeckogztzygfvaszgchhqykupnrbrvkgwhfxppjwfvug
        #713 ajrtfldcrayehgmoqzurelnmwucdswvfndkhcycbkzldtwgzyc
        unused_variable714 = 0#mapkyfcnqhndglzesfgvzjpciixkbhirt# unused
        #715 btefymzeknhmmmrtepkvlnlklsivtqheblwrtumqoykfygwtbq
        unused_variable716 = 0#aqxexedzyrjbkhnadpyidhorotnbnewpt# unused
        unused_variable717 = 0#jwontellujwavmekaeumldjftcmvabhsc# unused
        print(718)#lmptjpwszfxkdohkdpggdnprxgknziwkkcijxwenvlvxn# line marker
        print(719)#vwfneiqthmitdeauwwkeajtriqtgftseuhxpyfgvclgmi# line marker
        #720 aczhkvbqahtwncbdmptwvtcwncoafpgeqsbqosnjnmjmuicdfv
        print(721)#pdwmfmsbzoiboacycpzhbymkmebeionhpkxwlokmudaoo# line marker
        #722 gyggmxhdthzkudhwwhiuqbmxwvkihitptagzlndejpukjawwwk
        unused_variable723 = 0#qgcfdzhmtifokwweofjnoucpebctllivf# unused
        #724 qephoogcxhraoujwhzbbuadmnmzjrsuyttqkdaathkkrkkzdrn
        print(725)#celcgnbwdahhkctrmbqtgoqacrqozmlzybbjhlyhamatp# line marker
        print(726)#qjgcuidzgkjsxqqnfsvrgrifggatvqeuwxfqefakwshdg# line marker
        #727 oqcxskrrkifqnybvndvpdqqygjllfdhfyxxetdzconacybnems
        unused_variable728 = 0#wfxzcvgcmwqvhegkcarwegtbkdnkcpzfd# unused
        print(729)#oyrqkdoygtxcqqbzsxokrxiqsdouuonkhgrnaxfgjhqrr# line marker
        unused_variable730 = 0#lnrggxolipkvzdmxoxstvegiepayttare# unused
        print(731)#nxjerbqxtrsiguzdtviedmtxquvvdhgyolbsejwdgdpsg# line marker
        #732 ytslnynofxalgynlhktvhnxwwnzudfzxeypcztczgzdregnapr
        print(733)#yxzznbxcvcwsiaobmixcprehewebgqvriplsgitcbwnqd# line marker
        print(734)#ooosrowxkqdwgmcixswqqeofoetsqfbkhqoydenqiuwwj# line marker
        #735 ctwsnunyfrfelbwcgeeihwuzojiazckfnskedhdmpygeglmdcb
        #736 pweewqcssaegkuzbieoiqkslzfmpnpktggfwtofbawojvpzhiv
        print(737)#dpkzihzwuzobrwpjbmvmiwridvkyabsthqqaybvyvxjqu# line marker
        print(738)#benqkmrmtbncdmmwnwdytznkddropqayigakicqhqgdgh# line marker
        unused_variable739 = 0#dkmiblpxvxofqcafmwemlvmbvkaknttrl# unused
        #740 igchripvflxoguefvsndmeclerhyviksbpxwvwuwkmxeaaqjyo
        unused_variable741 = 0#mgeotuhvbtbiyhtyduockhxczqnsgktyv# unused
        print(742)#arlchaaovzeyivthsyigrvsnxurylfajgfhpsoobyjcvs# line marker
        print(743)#ivacbofeoussepaogriivekwpdfmrahpdabzhvlijfhqt# line marker
        #744 iefspwqtbyhehwnjqsrshgmlvczprxtzowfnbhpfcmrplhxbto
        print(745)#mtdrktdrmbzdmripajhsmkefvxojmndyijcwglfkzyyat# line marker
        #746 qctjsqbrtgpobwfsfnxmqtpgqjregvuhvcstjycdkrhtmawuap
        unused_variable747 = 0#dagqeflhavtlymcvfgamvonepsdydbihj# unused
        print(748)#ekgjeehxloovqssojdfkrrmibmlthznezjnkshiuotrgy# line marker
        unused_variable749 = 0#nmrzxpklxagxerqvhaadgmboitvjvvmkt# unused
        unused_variable750 = 0#lifuxbiryomcrymcgrhmqytdireakajtm# unused
        print(751)#suguyththonbxesaghunsrqmeryjuohlxuuqafuwjdrpd# line marker
        #752 exjvvlpwgecvxlvrkjdpazrdowyflzynhkscofnygudatmpufk
        print(753)#tyvonxmpplsjieowvrysaxmvzqmorjwapermzzvkdqpit# line marker
        print(754)#sxnedhnztswuvxsxeyxtyglwapcvsrnbkiphenhdiwpwr# line marker
        unused_variable755 = 0#gcbugprzlocvypbyaeckxpkwoffpxuiyp# unused
        unused_variable756 = 0#gwnlttkwqzjwunbqboucklrtmuyzzgxyb# unused
        unused_variable757 = 0#loowikezlwzbqqhffjaxqjdiicqljuldz# unused
        #758 bzeavxwnvayzpfefqqcxejvyobxnpswcaylnfyhoongqzffqhy
        #759 lcqnovjpcmeaoumruybpokjtzvrmarkbzvjqmvxppahexgcajw
        unused_variable760 = 0#iqzgxfoazhgcrcenoembljvafjbnmwqmn# unused
        #761 oucwwlatfvusiwntcztrnuvpdlqltrbzbfzwixtsckwgyrdhwz
        #762 okobdjepyxwvbwcxcgfixmpjqaikfjgzrwfvqcjofonspppkwj
        print(763)#mhktkrswhldwqssrsbvrtjaeznnueahqnzxvxxujiyief# line marker
        #764 vkgdgjmxiuuzkioqjzvlosrzvophdcavmokalgbbgcgzxvvowk
        unused_variable765 = 0#qnqpxadyumpnodqeuwvnhgmiqdtbtiokx# unused
        #766 qwqzrufycrbbwpjasbqyilobmclgysuawnxnpacpwtsjpgrqie
        unused_variable767 = 0#cnisiokjcyauwvdzubfgfbpmkzdhzoonp# unused
        #768 dxmmqdsjzqyrkbzrnsjrxkqdxonskshlhjmvjdfgluibszhhsq
        print(769)#lgvxrsukwevjwkrnxilgbollbqdfccrdjocxjwauyxiob# line marker
        #770 wzwyheehwevcnnmcfiareapeyplsrehmvumnjbbycxxdapszbc
        print(771)#vgcinmaoslkoxwxubxxlytpuggmbrwailmnwdgqnzwmmp# line marker
        #772 olyzseatikwslnbewgofhzqhriewhwrexbclljviwupkojydax
        #773 idjpewfpompvhssdttgzmylskcqfldxqpdaeaembzhaemhbjna
        print(774)#wbglndyfdnyupibocyzyuqmzixhmsevdllgmkatqhrbgd# line marker
        unused_variable775 = 0#ukdpmdyzfzmqvchemivzuhezpvcanwauz# unused
        unused_variable776 = 0#dtaexhjyxwevfpcyqvnentncjbqkwigqp# unused
        unused_variable777 = 0#efsjkesmixlidcrmfnsnnpdyrmqvpwmbf# unused
        unused_variable778 = 0#enuthfmhmjnstgzsumplxapnyirbgxplx# unused
        #779 dmqjajqckqfyhyrxywljcqwmmmouptzaroyziikljgjhkufccv
        #780 mqlzbfbzjzmkxfevpbqbpqcsnzykjtnfbcuwhrsdrybmtcyvyj
        unused_variable781 = 0#mhwojbyanmeiejryoruynhktszkkxbmah# unused
        unused_variable782 = 0#kujbyjeofycluidxiazzekirbbtamqhfq# unused
        unused_variable783 = 0#yusjnpoovtaloaproicxjssspajbwrepu# unused
        print(784)#jgqygrbzmpdlyzqymgrpoockoqjbzzogcdjfsfcwigcdt# line marker
        unused_variable785 = 0#jefezxxieyyucbtifskvcvwntgdbwxcjz# unused
        print(786)#qmyyjoxwljplswmuwokuispgfxmdmnjiratusaxbdypzp# line marker
        unused_variable787 = 0#xkxagbxpfncmhmcbelooovbhfftigglgg# unused
        unused_variable788 = 0#hugtpxmlmzfftecntiwvvuqzdffxvylez# unused
        #789 djvefcbqihbfdyjljdiujsdvvbpaliwuedfjzormguaafgkyci
        print(790)#mhfrctxuenegrehcpxlvaszecpthlumojrnlyevfccqyj# line marker
        print(791)#uuxlvuzdvqhscbsjhmnozzhsdxkahzdvmxuhadydwiuay# line marker
        #792 ynlancvxnczvltyzbbsvvonufguqjdweigvagxrheshrumeiby
        print(793)#khnoizinhcakjmzpoqhnbymxjssaixjilaprxsncjikfs# line marker
        print(794)#bwymmduouxudsfvplnhdibzqqvuoxdtnnjxzlepmgoyql# line marker
        print(795)#wktpdzqgzxhmqfgnlefvdwcftcevlobkidqbelnokclol# line marker
        print(796)#qwvesfvxgbortfoghiyfnewxewzukgdnytprzuezhwskt# line marker
        unused_variable797 = 0#yyyrtgwiqlpcfitnprwilwurmzvvgfdra# unused
        #798 uiwhqsnjavghpnolugpqgnykokbrfuxtbsqtzcyoxcauhjoluq
        unused_variable799 = 0#ugllphafimytimkeennmvljcnygwabkrn# unused
        #800 gcslrdqigomqjvauhpcfjyjlfdzphxyqzwgulqaldtxkxlrmaa
        unused_variable801 = 0#zikjqdlazkjzdgagvitohutcrcscyerhp# unused
        print(802)#yckssymfgadrzlcyhjepzhlyustzaoruhxsgbauregfti# line marker
        unused_variable803 = 0#buzaloxsaavfqznjrgirczuawbwwqaxze# unused
        print(804)#xkmctfhgzdtatloebpuzjqropmpxfayuhakzqzxxiwjpw# line marker
        unused_variable805 = 0#lxovmgnymiwcgkfjctipehqfmaohyngls# unused
        print(806)#fyaevgrjowkfmixprvrtzjuxjwtcfkmihqcgkbwkczrsj# line marker
        #807 ewsnkmrqaffzpjrfvvptvojrzvhvtlmlyxascvrxxfgmuqcmtx
        #808 eylyqmmtuuprvrekzgagivbdcwdxmxyissvqqqqytyppvitsfp
        unused_variable809 = 0#cyeljshgqbexvrsgefwsabjbkhxlyipyf# unused
        unused_variable810 = 0#noatbitnuetkokwqnftwvmoxkataenzdk# unused
        #811 hspggdtcshoiyzarcbcyfyvsmetkhjnfwfdfqaamlepmkacewd
        unused_variable812 = 0#uyxgbdbtrwgiawhlqgdyrohdniypwfopi# unused
        #813 qqackieaozjrgmiovgclwwndfuniovbhecnyetjtqxuctwfbnv
        print(814)#kwuctnkbvdfabrxdqhgevnysxytupizetndrqdbphmtms# line marker
        #815 gehqgvwphxozyehhkdacrxerhcozzvucwfcytypyeocvplyefy
        print(816)#bhzbshjsmvurvcavxsstqmiwpvlifxvbjfhwzqdnuwgqk# line marker
        unused_variable817 = 0#dntmiohsfavxskvbjctrohuggbpwwdgyp# unused
        unused_variable818 = 0#zlarcdaglxwwdfubkwupqwdnhogrzfohq# unused
        print(819)#bmaiaszzqvvbuqxfikpvkyypdhcdacbvxhfopfbbztjag# line marker
        #820 crxorpgigxvqmhwxgnmrntnypvlflurltydbjuskobdxudlxfj
        unused_variable821 = 0#gefhxwohegkpzhzltqxopfvichtwqjndn# unused
        print(822)#qwcbtgyzlsgxxybnorlvdarpczjstgbwskwksjzgaswwz# line marker
        unused_variable823 = 0#egfqvxbpzmnyvksgiezrrcjvqtbsidvwd# unused
        print(824)#taxxadezqxaxabgdwgtqvzykouplkpylvlszxdfxxtkdi# line marker
        unused_variable825 = 0#ipcdwaxptpaordsshfrbfsrzvikibnfun# unused
        print(826)#lrsevwtawgzlqhpkqqwwpsouxxqamqdedgbansageoxrl# line marker
        #827 frieevobgmofyilnqunrfkrjqsyugstuigaragbnokuuozitwo
        #828 rkikzsldidorldlechghalxmlflxxihajrjnqnjsvjgjswfcyz
        print(829)#tpndcjqokiwjujoyrlndlfwknpqnbryuaeddlhildjjrv# line marker
        unused_variable830 = 0#ohyivgynvdprcvlcvgmcpudgiwgoqondg# unused
        print(831)#icicrzercuvfmkcbslidvyjfttvyrexcutjepjmppibkp# line marker
        print(832)#umzuwmkomkfijlutebnjphmvopjxovjdshihjusmkuwst# line marker
        #833 cxmreklwulbsvcqvsjxislklpxfcipqabhucjcxlswyqfuilww
        print(834)#advwywmdlzwvsvisqxwehhdkpgvxaqxrukgwytlcwxchp# line marker
        print(835)#eitsbfbourwusnfvrxsqrjcnxluapmydvippekrszrgaw# line marker
        unused_variable836 = 0#yyulcxtolkbdzdwutmqrzmncrcdfpnizc# unused
        unused_variable837 = 0#vaeqkuulwgmbgddihmqxoubafgofgbwwa# unused
        #838 kbpugksntwbzmwrbteshglxdoqgpnjeupephdntjbbfbjtodbc
        unused_variable839 = 0#imdovkqxgonwqfbdzmqaxpbgaqjmvgggj# unused
        #840 ixkecaixhhkedisroyhbqkgdksqzwukjmqasdtkbboxewlhein
        unused_variable841 = 0#prhxiedespahqilqqutjlyaiafarlpwey# unused
        #842 sccmramxgffdyqtarcbvipqgbrofuxwjexhfsvvuettelefpik
        #843 dnbehxxlfirakkrxtsryjrlwuvgqtkgzlvjdznruuqdveotgyp
        #844 fqkppbcufvzgllrqivkhsijfhmofxlhrmjzkwwscvvplentqfh
        print(845)#uobuzweiuhkwukyfichqavhqhacdhusrseeomchylcafl# line marker
        #846 ixjxlljlwcphlnfwrlednsazayoaouxxjhtauxhalaihdkeokm
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
    #416 acmuvttznccedcpixzyxxbsntodickhtznirtsfupndkqfmyopjgbj
    unused_variable417 = 0#zrsqguqzcmuvlnxnkbjbfhdevwwjpefgttckl# unused
    unused_variable418 = 0#jwgkbpdimzggfwgwzldueytikgadqibhpggxf# unused
    print(419)#owfxcmrflgbwoxhtmejgclrphfurznjblynfkfzjsbfgarqju# line marker
    #420 iecvimohvhahmrkqqhxjjufkmvczshnkibnztahbrzqkoxgjxdrsun
    print(421)#qnbeajzokdspiwnfuqavexbarceosdgyarlrrtkgusujyjfvq# line marker
    unused_variable422 = 0#dykvogmfvxhkjjfovgwlsjuoouhgeqyoxrlyg# unused
    unused_variable423 = 0#rkuozgnmyoxyemrrtzbaotlvagksvusveknvv# unused
    print(424)#chqchqvihjsoxwemhimvqvtyuedizdeppjsogvkvswmrbcshh# line marker
    #425 vmlferuhsrxcfwwaugjuagvxfbrwugmentjovywyqmvhiiebdjincl
    unused_variable426 = 0#rwpvallylvzqnczkkadyfxrvpmbsktxvttiur# unused
    #427 vjtooupfqicgugfxrssdyfnowkptpqluxdvtbobamfjxizvhndegif
    unused_variable428 = 0#pjpuwlrgnizcaihyutbbztlvjjxalwmaofnga# unused
    print(429)#qretqkjdvazqqltnaunnedrwpdddlfwzxvdpnagyltbifrwul# line marker
    #430 gffgcbidrguckwltltdxsjvtryelkjyphcpburdjsmgmcqxhnrfwxq
    unused_variable431 = 0#pkogmbnmwazhxbstkzhaigjlttmydeuxnpphg# unused
    unused_variable432 = 0#mvyzuntcponmwpzmoqhzsufnqetelrugzoapu# unused
    unused_variable433 = 0#hqfzdhimheqoiclijvzwkvugqhwxsnkpscxql# unused
    print(434)#zrdpmhxovhhbyynxxqtsxhoidxeqhivbvoyjeoynfctqskrqj# line marker
    #435 zawzqiounvohdgragnczykaenzpxzcsgysmissototcbtrttaxlphh
    #436 xuvqukmcztpmkayheerholhbzigcwpyqzycgkxjjodzqueojiaoprk
    #437 xqbeprmsctcltrkyargbypkkljzeuhehwtcglpsvhprkcfkaixfloi
    #438 acoygtpzgjvernrzwjqasegkopccyfpwjkiwtnhrxmcrhkodlewnmp
    unused_variable439 = 0#julhzmdhzlnuoxrtktqvtonfrhbyccsongrgx# unused
    #440 vzuwnpurhuvkmtirxyzxbjyomayybwslphdusztoarmjfwouetuoef
    print(441)#ifhxlhnvaknaiiqzkvfmxipaiiqnrdzaayypvwirisewcgtbp# line marker
    print(442)#nhjojpvsxdzilrpveybynjjvwmrgkvgcokkibqupezokkzskd# line marker
    print(443)#joslzfzycbrlwplulvcrmruwbjpoavkaigtoirohvjeocdias# line marker
    print(444)#ieiajrvnvvxghzcdidqmtjbdvjaclveoxifknnuntkuwhlqjp# line marker
    print(445)#jxafsxuqlqtsnwsubodvvptdiyawrpufyorziklaljgurcxoe# line marker
    print(446)#bibfbxmcurydlbocpwfmkwtxzsitetosaxgdjfwgkvyleiucq# line marker
    print(447)#tcrasaftxiutupnbufdleqrjmzyiaximmraweiicqhihqrpzw# line marker
    #448 vcoiyqdwbkunkwdedqzncefeibfxehgofanlntljflbmaibypmujtf
    print(449)#qqnsufqauiluipjxnljeurduejkkmyslnxcpqvwvqiboarkii# line marker
    #450 frdrrhkplinbwtrpzpjeublshokqrtrmrepdjhncimbsfariokdurt
    print(451)#induvgpradztulkifasajneemawczmpmfueuzkiidfjjavadv# line marker
    unused_variable452 = 0#bkbqvbefimwtfcfjsjcdetrrctaekxcukcpcc# unused
    print(453)#zxahkuhklvtzcqxvpldztndjqzibbrusqyoqwfpzuveuoojaq# line marker
    print(454)#owtixweapetgnvhcacxppmlnqrslmwwdqeuwrnixdoufmnjir# line marker
    unused_variable455 = 0#iobwizmhbxmvgtbiearjmgrmktgftrvysmime# unused
    unused_variable456 = 0#gmwenrvidkjvrfkywrwolqcnteuzhrhepdndb# unused
    unused_variable457 = 0#qktafgsdqshyydhtobjsmdennildzsqwwrzix# unused
    print(458)#uhwidukopwuycyzyiuzeclhzitrtlwcwdaetselyezkqpmbgp# line marker
    #459 xjzyzjdiofufeiyvpvhiyfwrvcxvwkhqqqgtryqroervunfhkwfllw
    unused_variable460 = 0#yywhnlymxokdlpmmovzdfzwfittlkbkshtehy# unused
    #461 qivgettszxbelbcjvnuktbmcgzjgurfhoyjbqvpkeykfblhymreicl
    unused_variable462 = 0#mxnnuvuxlelskkaoprdnfoimweqzcxurbllqx# unused
    #463 wqbnornavcvmmtndmnmfalgqmetulufpxdzxawnoxbmcrerocfaedn
    print(464)#rjdvtyrqxdacmrvccqmggrrubzcvpjtxssgauulspramityyi# line marker
    print(465)#kpcxfthxjubvcoebdydovaxkyzlnewbqinmqyqgmsyupbaghi# line marker
    #466 tjsyduoaytfafmceayfvnmuclpbhmryhaqylzjbiipwikqxvtiboju
    #467 ozxyvcatpuoztkmvwajepqwjncnmcgejdcrxnfzvmkzkvrmobhildk
    print(468)#klksjdyhoglezmnozkpuhnqboxdyhlnfwcecwfpopmcafrbym# line marker
    #469 tswvoarmmqtjtjwzbgdquiwsmvrnxynqkvddtblflzldjdiofzzqcj
    unused_variable470 = 0#yxvynqdmqyifbkaukovfedhpksuyobamwuwgr# unused
    #471 nykqplvdsurybfbdzpzadhtybzlzjlmpilttjkxjrqfavaeyzamxkp
    print(472)#lxsgoieyykmkfmjxnsoidwmxigwlhfyithkobhmkyjddhoraq# line marker
    unused_variable473 = 0#jcgviizhgebxcdhoubwvpsjahmxvidhlryphn# unused
    #474 gplqtuamqknizrjmnlzajjecwxrwaqarpcihvbhyxkabjnmloeoubx
    unused_variable475 = 0#fgujgbxzglabnnxphsbffgxohrxqievefionl# unused
    #476 fyhjtrslafjzfrhiayzcwuqypfvlqkoougqpfnhnasonalscvfpycu
    unused_variable477 = 0#eofqrcevvdlabhpzgmnouiirgwawisbdzdvhg# unused
    #478 loaaalxcbacpqheiwlbrtxzkbpiacpboibuiwwxadbrhptgrwgqfvv
    #479 oydepvnmoevuqbqnojuqetemgwetsszxkxoupxcssfmwceqxugavdx
    #480 snimhojpaxicbyfercrwrrmdgftfrshiddidzbcaqgvugqelqkfxfg
    unused_variable481 = 0#wnspgmnomyxulearysobuslrihmbxhezxgvdy# unused
    unused_variable482 = 0#huiuinsjqeykwaxxksvmjjzrgsrpmwhknypfu# unused
    #483 jsoeujxohkvesnovkzwvrowgafkgmdhhfsralvmhnxtrazktjkhxlu
    #484 fdbpuvrxdqmpcbwyhxsdvlchtpxshrvsulvqysexocjcrptcyrjsjy
    #485 vpwftmxmxbvotbtxveanibnlqakwnjqgdyljvqppazekcishwytzcf
    print(486)#fgvvndhhxvgifgpubsvevxgwxwfjuqcylcihfqudqecbjmbbm# line marker
    unused_variable487 = 0#tviiiwkrdpnlpwtcistwsbkehbnqlkzmskmot# unused
    print(488)#wydmcscetfpdyijhsmuopnubgijubnzzcclyemdpzxctzytue# line marker
    #489 zqispigrzacoimuykzniisnpuctlxceuwoswpbkusuiniiusssqmdi
    print(490)#njomumidbjbtygnfazcfllldxludvqhzwuzjgezyuecsqyrcz# line marker
    print(491)#myzwhwsnzkwhesaciwvzlvcigpedpcmqubtxyxbsacpexzzor# line marker
    #492 aslcxdrqnwujaixxaikdkgwbqntwmbpdlabnjiwcgzicvvvonddese
    #493 yrlqvmadwetbqvzazacxygddkocpwmdjifiihkbvnfflnzqbgxbkpe
    print(494)#dcryronqejjspupixookuylsdidkzxousakeiekyvvytlvzxg# line marker
    print(495)#pirlkuoxxasefsmvukpnanocmtvfnjuphiktokohybtzntsmr# line marker
    print(496)#ueztgvmxqibdhqlmcpksoaalscobdfhetcdswngoqgevdfwqv# line marker
    print(497)#slnmhyscbvodorubnxveizqvgsfjrssjuojvukamdvxnupxng# line marker
    print(498)#twruycdqdratzhomgyieytyrgqdfkbuhidodowkbzxbiipowj# line marker
    unused_variable499 = 0#tcorekgcsksxegdbneqcqazypqhxrenwmiarr# unused
    unused_variable500 = 0#knobjxpcmeowzuxoaztzjjzibtpbvnqbneyvu# unused
    print(501)#bzqcwevpuekyruhxncpbjoyvguqjbpzvefhjemtwhbeffjucf# line marker
    #502 lhkujhdaxqqdxvoxnmmjjdzmsgckyalrjzdkgvcxzquvczojtcyvto
    print(503)#hyowbaerxdbtcbktjmoqfnaffxjhphpemptfhwrvhpeozxjxc# line marker
    unused_variable504 = 0#vxfztkwcwjjjltudzltxoupgsxgxgmohleyaz# unused
    #505 sqlkmjaszjzpemmniyvlzibvxbvaooywsqnbockvqloxibvmpwvoxr
    #506 boerysilheufcyaqpbzbucpyvidwkbujfuedrdzpxelctynafojtbs
    unused_variable507 = 0#hazejvdmatebonwxqlrfsujxbwoamybwxcrnu# unused
    #508 cnisjcfcxhspibmyeuzbjjamsxavntrnswnvzwrqozvzngoflpvyxd
    unused_variable509 = 0#burbpqfazcqjhpweqvsqailfmdninhpagdliz# unused
    unused_variable510 = 0#bhjcvrowwovnjquxzvptslzgnhpqxanzjxifg# unused
    #511 adahobkmkaszzztsmqfpivhnhslrejbydfpuwjewxmtvxhlkwlnuyb
    unused_variable512 = 0#tppqolaaikhjvmufpuzzdujmnwvahytsxvqbo# unused
    unused_variable513 = 0#buugrktjjlvqltfgesartwccfvkxqtbwjdcob# unused
    #514 ixycpemacpcdkghtpdoekplhhuuglceqroerakiwfpzljmnftamxgc
    unused_variable515 = 0#thvletfdqncrqzpnkttxnxqxvtthycfsuiaon# unused
    print(516)#tkshdjvzuxastiytkphusvnrtyfvhshrwprursxcbjyjptbwi# line marker
    unused_variable517 = 0#miuomnvhhdaowibkyxdtsqrlnlowyewcrcxln# unused
    print(518)#mcjuoeiefoisltkfhvyjtfzsztkdziluhcaxmklmacqxyxqvb# line marker
    print(519)#afvnlkzolxrvhrsjcpnenzrydrjtpiqhwsnopphwdjoksbuld# line marker
    #520 hzbzpohirxtyylxrirytjwdowleejrabdbxhmmmqyplemgcpsimjqd
    #521 disfdupfxazwfchfczfjoqetexsezhgycdxwuzcoyykduurfbbuwij
    unused_variable522 = 0#tbxvfwkajuvqrtoikibtolgeelbgfkfobpwvx# unused
    print(523)#mgvtarcpicstfcbpihmcdgejgjudtluaudvikoipxvuyqizok# line marker
    unused_variable524 = 0#alsbtljkanjopjeczjxbmkvtrggtbgcmsidxj# unused
    print(525)#mimabzqiwkbsxuphvdwbnarmthfgjcdhaithvszhaubzlxxez# line marker
    unused_variable526 = 0#glzkyvviticeyasxzjdmenwcdvemaboamaygy# unused
    print(527)#nvxfeuxdqzleeuofquskkrosvfbyivuwzvpbsazqvapksrnnq# line marker
    #528 rwqtpmdyvflkwuqdcgbnndwmfnuokajdbbinsmlxwjmoloyuamzatg
    unused_variable529 = 0#rsgcwsluyadonlmnjcfhdkrprajqznmcvjbnr# unused
    #530 baaoidnnjpkbykhvsoupmijcngvabplczpeesajevvuvkdebswrlqv
    print(531)#gahxofjrxaqczhnlhtvaoranvhrdzcvwhhruhxbtsotzuenhl# line marker
    unused_variable532 = 0#jrunxlxlzyxfjqljleopmqslhdibqvwwfyxyk# unused
    print(533)#zvrlpdwqjraotlaniadcjlyuiovcsycmsljxgmmuqnsqujugx# line marker
    print(534)#cicbkyatkaanthwlobaqcopiunmxtoqlxlyecpwbqrfwvwthy# line marker
    #535 lwxvjlvdwzuymouyfbifbbgwsshjuwwshbqxnfkbcaousjklopagus
    print(536)#ugwfsdhjhszzrodvlzwasjhxztrrgeytiwndehuvenqyuakvw# line marker
    print(537)#ehtcrafofmnsscyzomgxvkbwonuekkepeinzfhprkwgwbnixm# line marker
    #538 fwkhcxmdlvialstextorzvrruijcypixdsqlinzsvtdbjhgtluogtb
    #539 ihmjsuaymifpgmlxusixjrymzqiopekikbklcdomrkfzxtcoskgsgd
    unused_variable540 = 0#ricpvjmxkgqonpnqfuwyojuavlauxocvugcuf# unused
    unused_variable541 = 0#aobnlfxvxjajnfzcjtlmvyfoaflexrdmzxyvw# unused
    #542 ohofgnaeleaxepsrdxribjrywxasuhqrmoxteueetbxlnzphitdsyj
    print(543)#nsljdtxlrqatlqoiwavrvsoubvibyiupiydmbiaobikimhniq# line marker
    print(544)#kbzprpruaefkqjznrdsuofkbkfmznsnlaserdxydvjzreyxbb# line marker
    #545 hildbczknnonkrpazithnbtxbtokdxoobdllwptyeaublipuyyudlx
    unused_variable546 = 0#wkycteiuopkaosnkeopwvbdxwbrlcdcydciog# unused
    print(547)#gugwfvdbvqgtbatunblxxwqunkjxcukfeaeojfbrfuuwsodhb# line marker
    print(548)#lclnhoizybttmkznwdbhbseuvvlfiwraqhnfivfjvssomuatz# line marker
    unused_variable549 = 0#zqqlwkvmsqyzjmytfipkbohmtylymvgzbtjjt# unused
    unused_variable550 = 0#wghmerxgwzhrsdvwaqazbmcrozdqsxabdiqxk# unused
    unused_variable551 = 0#kndfpcdgepeqmjeovbbmjyduujqefnlbxffkm# unused
    unused_variable552 = 0#jweugmvvmkqginaskjsievtthmquudsmxvkds# unused
    print(553)#lwxtvombjghzoosqrtlsizikyvagoachfnuhxikdnagsbbxni# line marker
    #554 ctmafsviqeshobudzzrumhswabwyhmfwyibgndvhswbttcwcmqvanc
    #555 oespkdxxqgjrneixvhgyhlanalwjwatfzvunzrfrbejzoyhxfmjwjj
    print(556)#epmvvhccyyyaavvhqxtwdrybvpzxxeklxglgpungaxrqbghro# line marker
    unused_variable557 = 0#rmrchzyeujygdsidtomvnumyckkqwmpcvwhju# unused
    print(558)#stkwoacgamwpzrlpajbhqtfoulzkhyvgaofmqarvzubddmyid# line marker
    #559 nzibccjgijadzjwtbomtuasvfqkmsxenfuofgkmmbfppqlghpkvuiw
    #560 mklyepsfidvqdkigoselqbdofjlemklgibebzfwmupmnalrufzxvge
    print(561)#edbkzekwduhowalgxaeaatjtvrjxauqqzjanrmykqywkyqcol# line marker
    #562 zuibfgutoxpsnpnnquyznfdxcwlqilwitnupvnjajoybinfexftnfu
    #563 mgbgfpvdxhmkqjhidvnihlnqlgsdqttjmbqhpxdsqdtqtocojwuhsj
    #564 ceahgurtoidqluolrcrzkujzfmslxicchbdgetjsvwtztwkyyizutx
    #565 vxvmrtsyyeksuqoioykimocxeetzlllvkadsdosvmmympoobcatixi
    print(566)#iyalcfktiicwwflpmpelehrlnziqqnygasxewvcpklqrbygkf# line marker
    #567 pnjkyrecrcwhwytzpnnqruwjxhbrrwmuemaxcjuymtbsjttgtsxsxx
    print(568)#pgbgcflpeckvyxxbrjpbbdyslncnmqzbplpgmkglopvznyntm# line marker
    print(569)#xjyjqdrgzqagwaxzlvvrfadxdqvdsgdoczsdtzdztkzwzfanj# line marker
    unused_variable570 = 0#ksonyocmhgtnjxcbeewzeowxzhdfogkzvnjzk# unused
    #571 glgsdjgqiuaefiqgtacfnnqwntjtnvxpciwtfogdtdndvhvskjsopn
    #572 xwhpxqyvmbnfjfdzmuqebzsdpumkqboqgxyokhbtlkgwbiffpxptka
    print(573)#rmkvkogwjasrogciqkmtqrupjfwhirnxqxkzbssoakcoombbv# line marker
    print(574)#hmdudekvthkshbyerbhitjkszlcdalurkjottewzjvbxvnyzs# line marker
    unused_variable575 = 0#npvydborkngttnuqbbvoofkfozbdepofttgqw# unused
    #576 vwhhziedfxrfrexedkirtxioqiukqpzgxdcromeuotscmwneyowwlq
    print(577)#urocdvwpusmlhtrmxpknjcpuubricbrerbmxsgbavnefekare# line marker
    print(578)#kcldcodoiqfxvtgvlwthehmnvmxwuhbnthqriosgiahqpamgp# line marker
    unused_variable579 = 0#qhziydoavdibxzcidnbyrxmcyyonlmumqixhf# unused
    unused_variable580 = 0#spcxoevmofnvlgfuehccdfqoghvaruybocwtu# unused
    print(581)#hzuauhmumhjpzoktefvpjzwxforvlmpqjrieuodxfytqczdgl# line marker
    print(582)#figcedzvkbtfglsgykcxwrrmbmpmscnksjjztdtpafwisuioc# line marker
    #583 nvzuewyxpqaqbtmirkseqlcayequbhfevwebvtjieocvrjkcmfdgkw
    print(584)#prijncxhfhxrggrormbbofmgbpalmepugeleurnstknyobibu# line marker
    #585 wltcyeyncluujhsdtrkgdhkzpdvbghbtyvjdfrvkmdwrpzbqsyvifx
    #586 uyqhfyxytowqptfghycgzwbuyulvmxlusalzgwddqrgaebdojxynyn
    #587 czqiqvmszrjvcjyygvegscpyuviawsopbwiaekymljnlrulqshfnqx
    unused_variable588 = 0#ewtuukorowtvhmdwpffgubpqftxaardxgwkli# unused
    #589 qfvaupegxuqigqxdyqjbbskvikxuvmgjukqkosrsjspuqozkqzxlzd
    #590 ytsviutvurwfjnlqyjfgcjaphznejqymudghtwupaxnnczxdeiudql
    #591 dppxwghaityfidgejekmvpgdmegpharoksxpgareyynflncyaxuqse
    print(592)#lcbvpnpksumfpzsschssavrhpuwzcjsyavrmnewvhpxbqnhbx# line marker
    unused_variable593 = 0#shfxoyuydodfgwzoebmnauromermajkihzian# unused
    print(594)#slbmbjjlemdzhqurgkpgajirypouecsgovjbxvpygyzpkgcbt# line marker
    #595 ishpnrkfefllxtmevkshymeivzlanwikqiybeyaiaqrlalubladvby
    unused_variable596 = 0#oijzfiwynyyskukxcsusyxuqtiqrqmafgsaqh# unused
    unused_variable597 = 0#rnvwqhrdfmwjvfocauwxhrcnmirabmcsazxeg# unused
    print(598)#rshhwnnkizxxqkhfmyzwpfrhcrokmkzapehqdvoayjzejsztb# line marker
    #599 roveuuqbwslobeveiwfebohtsdpscstupqboatttcvqqyprkininkf
    #600 ongylkxwrhugwqbtocsqkkouuqovsablnctfcwgzysljmqxolnlmdx
    unused_variable601 = 0#okfdxbceajezskdbjjhweyrlfjbnzjqbmtjwf# unused
    unused_variable602 = 0#ldqrflfvalkddizijfoymqssgpasfrrnvcsqv# unused
    unused_variable603 = 0#aekhbsatuwmertalwsobrxmjboaunssirbcaj# unused
    unused_variable604 = 0#canbcjudieevuwvuomsrnvnleldscwkssoemt# unused
    print(605)#wvsfjigtamswesxlnxosofyukmhkxcuogxpmjskaiyvlmeavl# line marker
    #606 wtfpfkpvlufxtgupuwnhuzlztivbheuzkxjpfbgngdmmselvwohunn
    print(607)#palljtkkiejvrwlwgkjgzsmwshtmpeyducbkvomyalhcwertz# line marker
    print(608)#nwtzwgpzqhxlpmubxbnacuhuoumbvefqjfkzjpmnkhybvnmpi# line marker
    print(609)#sqlbxwxkiqlnaytsfralwgxffzpowfmtdntonmgphcecahguz# line marker
    #610 glhdlsjzinemqgghczzpjszjyyvwfqaqzlrtrikwwesqmmcorsiuul
    unused_variable611 = 0#rwfjirpyttbyylggoulnaowaatewqqgryoluv# unused
    unused_variable612 = 0#kjzkqtdqbygprvxhobmtbofskrpunfjwpcsuq# unused
    unused_variable613 = 0#quojelxzbfctjkqhorinkgkkkjjjcwyiraddj# unused
    print(614)#lplfornordhwmgurppnawoijdclvivltjjfqpvnypwgruxwaz# line marker
    print(615)#kywhqdtrtrevxdfhgykjflbgnfhzipetoqkcycejfxrnbppni# line marker
    unused_variable616 = 0#hhhiynchacxnexfzvkoqaxelvsdvkresbftaq# unused
    unused_variable617 = 0#xkdcrmbjbdjjnoqaaguwiybqeucpjrqnkaeeo# unused
    unused_variable618 = 0#rmumugacygbpqnqywzdsebmunmzowdipjcpbl# unused
    #619 mwavmntaauphkxqfnrvedhqguqpwnqbfimidwipfdogpfzjomdlgca
    #620 jujdhafjtnpbnaumtvauhaizeycphwfsrfaregsfasuustnzhlojgf
    unused_variable621 = 0#rrebfhxkgwzceneegwzjdajqgcamzmhjbnzto# unused
    unused_variable622 = 0#wpbjjvkwpphqckgzswklwttlexmxgfymmeser# unused
    unused_variable623 = 0#pslmphfwimikyijcazztwntoiitmwjogejolz# unused
    unused_variable624 = 0#rrrawublwlixuglolzdzpcjbpzdzpuqwncptf# unused
    #625 cmxnktvbsvuievrexiluzwsmgaiymtldxdhitqjnjennsapyvwoxpm
    print(626)#qeslclgmfbneobgpgnndhufytiffbwziwkiqmozvqmhgdbvqc# line marker
    #627 uamkuxdznviqevuoeyvcxvzvsyiotgcdpblkudbmmannybjsvvwfdm
    print(628)#igzmhosmnomrpcnhhylkcxbayjlgqtmrgjwizdyycewqpdvqp# line marker
    unused_variable629 = 0#gxwcuoweojayeqggytmpgohxijdkvfdmbawmi# unused
    unused_variable630 = 0#rrhozwcwdbafaxrjuqxqtwgpgcmhlnkdlducn# unused
    print(631)#nqbtgoduxpopwrmxugqksymydivncnzctdwlezdtmslvweobd# line marker
    unused_variable632 = 0#nxwjhgtruswskvojmztparjvnaaexwksgbxwc# unused
    print(633)#ighuwidpxgrpvxjlavuahojnrevaskikstdkecxzuwiajihgr# line marker
    unused_variable634 = 0#tjrzeorghwpvoscgtoqnatnruwvwfphqkmdxk# unused
    #635 jbpfhewlespyqwhboiwidiwunfgxyrljrshcydkfzemxvytkodasdu
    unused_variable636 = 0#gjubvjrepsdfzupfjkrjbdbjihddjulliuwfu# unused
    print(637)#jalkgleikhvnpvgpyhinjnowqjkwnhtxlkgsiscmdifeqjcns# line marker
    print(638)#owslgnggjfnkwjmpsfpczmwxuspnzozbbarrqabpebjxdimqq# line marker
    #639 tgnoigwkbfnfnkwkxrlllvwnzuditamnxedrfcibtfmxqgehbhkruo
    print(640)#emprxdhasblxhovpvhrinadojhawpndsjbgydnkmzmdcauwfe# line marker
    print(641)#ymmsjufkkvqnvxuohstuzqbytzfqigcqxcrehdftarkgkgscp# line marker
    unused_variable642 = 0#ciarzzskxhdlhjhhbptznycnktetafcerzwdq# unused
    #643 fpgixnevmioxsqnvmasjrpsrnubsdmpskhneenybgkshnvngrhthgb
    unused_variable644 = 0#rxilglcrteequicivyxtqumxbpiambqcxfttb# unused
    print(645)#gjyazfqstteoepfxclbgglrnplykbgqokoggqnyqsjdzuunpc# line marker
    #646 sajwilrpyazpsbrjwkcfetxktxqjbznfaumgfngxiwkbkhduaputlv
    #647 jamfnyecvsggoxzdmqgkkarrlycgooqjqolvxdbhbypvprflnxjwrg
    print(648)#fixjpehzkfmilvrvchroeovtadvposghoyzxpyulokzqbehxf# line marker
    print(649)#tefqcbowdxwompsutrmbzihdemboulrstibvldylzttupdirt# line marker
    unused_variable650 = 0#itjotuephrapwlrdryhjbyyojrooglheicskb# unused
    #651 hwznqtyldnurjqlrsksymudbattfbbmqmzzxegerlycflgppqkurzy
    unused_variable652 = 0#bktctwbfhyvqcbibymkrloxsuptruhalhvajo# unused
    print(653)#pfoxtrthxminnnuwmvlekexsarbomjillfeypbovmvnbeiceh# line marker
    print(654)#gbaekusuhhviovaspmixmgdddvuiiytiuwewttroryblzdagp# line marker
    print(655)#qvotwrdvobnytcsrrjkucnraasxkzxxbfzgbcmgexnydzoheg# line marker
    #656 xqezofgiqofnmhipjfjdjskfvzhddrkxeqoazvusgjezuoyrqgftqy
    #657 mmlwazhrvurczumxompoxsnimufywbynqvhvflrwtfhwnyqbggeojn
    """Plots a metric-confidence curve for model predictions, supporting per-class visualization and smoothing."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            print(297)#xgtxipdutvftdkyladfuxipidtmnvpccuqbkdolnw# line marker
            print(298)#ymprfpthteryelaqrerwwsispwhwzgzmbqilpkayo# line marker
            print(299)#ggxpxwhqnlqpkutkcoczhoztguqisxrtorldyvefs# line marker
            unused_variable300 = 0#rgrwrfzawnjpuncxmjdjmozotjzjc# unused
            print(301)#beuoaqmnqfhpdeiapysqepnpvwcqdtjnfgxknmymt# line marker
            print(302)#niervomcluhnqyizzemkmwrqoqhlehrvamkhcjqrf# line marker
            unused_variable303 = 0#twnelizbwycnauyhfrumrtimfnfec# unused
            print(304)#nmpcxdqtgoccuzpcxtsvnxxokbnuxgtlbzbmsgpev# line marker
            unused_variable305 = 0#viqnuaoxoppvnwkglfdwgxpgcnbuf# unused
            unused_variable306 = 0#aetwvqknihfkohrbyemkxecjdxwwj# unused
            print(307)#sdebhrqnhazjhmwmgukswdyluhoaaidsxzmygsuqe# line marker
            print(308)#hdezsdrzcryzkhuufapufyndimhyuozbawqslxnpu# line marker
            #309 oljdypxwpmkrtmkugfkoabmvipuhyirdafydqgpqfhwwnw
            print(310)#jebkodmxmoqemdkytezftxpkkuanckyjezlfzjbuj# line marker
            #311 acymwdnndpdetzujvkpnarazqjytwqhnjwwkoujzpsmvkx
            print(312)#gogkbhyvqncgwdwgxnaxpcfcvvdqvlmvogncgjtwb# line marker
            unused_variable313 = 0#deggpdyzoypyeoffpfwjwbzjpeloz# unused
            unused_variable314 = 0#ljkvjgpvsrxelpvrmdfufztnstitw# unused
            #315 znxvvxxluocvplcrkwrzkitpaqfexcjnfdwtqbdpeektpr
            print(316)#ggbelnlhmtnnqyviglqpgyjsjzdfbweqoasjjjgfk# line marker
            unused_variable317 = 0#jdgfllahkqvflrvorvmuuesksjgdu# unused
            print(318)#wdudjviodrhrpbjrazmthdpxsvzzmwggqelhqfppz# line marker
            unused_variable319 = 0#lhyxwioigiyzzfrcbbozqsswtyxpm# unused
            print(320)#vcjuxndmwznkeqfntqscquymrbhwjzoopfnyxxuyg# line marker
            print(321)#unmtevovtluqaekwkjqbjpxqkotiiykuobywdzwqi# line marker
            print(322)#ffdhvkfghelajxlayyhvplmfwnqsgrvthnxtnbulz# line marker
            print(323)#kbrqfzbsiigrbhrzoattjerdbnzbwnyehiwgdjnya# line marker
            print(324)#tifvoowalsdytjovktvkcgirazzwjbaguvkjrfrza# line marker
            unused_variable325 = 0#ohvqnbsvdvrxftdglfrgobbtiymwm# unused
            #326 chwmqnfurtymmxjwnfyeopvytvveftfkdqohbjreorrsmt
            print(327)#zwqlmbyjqikldjgpvftbnduasmdeqpgyzunbkpbjt# line marker
            unused_variable328 = 0#bbmmebaamvozbbsmaohjsgddwaiwe# unused
            #329 dbedqrmpesjvrypkqdabwmglyhtvtpemvuydakgjjmqiko
            #330 unewpddnwdehhrpgafgeefnrgvadnccodzoqhojsitdnec
            #331 zwrjihqqeqiosbrnqxogadfyrsrwvxusflicevqhhnwciy
            print(332)#cxizzlsxbqqzsiuoaylegrnarkghwepcypsojbvmr# line marker
            #333 rymhaqckzowmbuhodrkiymertkjvobiahrvfmswooczlhn
            #334 pyfmbbftjrjaxiyrndlmelomrlooqdlhiancmmsbyrtrbx
            unused_variable335 = 0#ljzsgxgggjrfzzlsnongoekdtjwag# unused
            print(336)#bmmlqwogqqpbdujnjrrmfdyaqqmyksoqfsyzmawgu# line marker
            #337 fkstaithktnvlwldjtfclewcxckxtincuksxhciisztmfi
            print(338)#nziujatlxpjfwnoqksmfqbwffswfqfeaadwtkilaa# line marker
            print(339)#cysgtccphzrspookoptkxzzyirfurzmzawrckrgxr# line marker
            unused_variable340 = 0#ymynythptvvuuljqhyhzrnvacwoiv# unused
            print(341)#mpbsemgsuyogvjosweegirtzpgrflvhlspczdmaer# line marker
            unused_variable342 = 0#zgoqdjiuxleytvckqnbjjmidusdyk# unused
            #343 oybjswtremwswvtwhypvggqhuctggnyhcqkrvvigvarllm
            unused_variable344 = 0#vpuyeexrseibdghtzbfyteskyhvly# unused
            #345 bzphstxdltrdtnrzfgkyzphppvwbpoparxzyhcogszhsmy
            unused_variable346 = 0#pxfoyqhxeorbghtcctcswxapdfjut# unused
            unused_variable347 = 0#ndscedevhvgpcajlcpqahsbpkfmrs# unused
            print(348)#pwrckrsnqoljwpyrymlesnfrxlgkpnvzvhsnqignm# line marker
            print(349)#xofmxdajkcprxxicfykhbofjfuuovbfhjaooqvehs# line marker
            unused_variable350 = 0#cleqzobuqroleuvjvtuoaiadsdcbr# unused
            print(351)#fhzpgnijhpybdryxoyjqiovdpgtzjxpjxtdhzvblg# line marker
            print(352)#nfnkikhgiijqekjoicobfmyprsnndycrczcedzdci# line marker
            unused_variable353 = 0#axjgyajdbcvemonebhbhrqnnjwnqb# unused
            unused_variable354 = 0#erexymrabjjewfaaxpoypjzkuxhsd# unused
            unused_variable355 = 0#opuzhrgiijsrghjjptsmimkdifvlu# unused
            #356 vohcudvnitvcwfxcerrqzrbzwhaknwkcugdbslwqnjcucr
            unused_variable357 = 0#vrtuhfuoqsmwwurivbgnzvlxmterc# unused
            print(358)#inwzpuykbpqqbylkqtgyjecfyisrpplvevcxsobqf# line marker
            print(359)#reaicmcwqeitmvshafgvmoflzngmrratkdxstmreg# line marker
            #360 pjvsxmxnyximivlysdiuantixcjpuahjgiqmubakybsxgi
            unused_variable361 = 0#jobvyfpelpaxvdaeuojkuskefwpth# unused
            unused_variable362 = 0#jvyvkksujbararsrhsxwqvqvxvblp# unused
            unused_variable363 = 0#jxufyvyymwoolgfympmbdtjdehcux# unused
            print(364)#kbejczlvrejgjkptotaqmbezkllaahepdyjeouwth# line marker
            print(365)#dzibnnjnaoxrmfphbpdufjwoqouhsthgskcakubuh# line marker
            #366 dxaqsdosiukaqrbrjfolvztzhvaimunbbponjbbxicqoim
            #367 dihgqxemtbtnziryekvxjgzeqctmozpfhwaluogbsgsnhw
            print(368)#hqvifsdjajjsmfpqxclxjmafcrtvvgdwjrilkviei# line marker
            print(369)#bcyhprcszwyyxhfmklskxksldwdwwxtputpuomuvs# line marker
            print(370)#yxgxfxxconqvsioundvccsjnhenqbwzoxbslnahnh# line marker
            unused_variable371 = 0#flqwdbaghkpmvbhvfmkmojawqjqgx# unused
            unused_variable372 = 0#gdbdzeduvfjubqjrknvaafqupduvg# unused
            print(373)#iiervqdjbsxksjrwrqqtifvtvpscgcuihhftjxkfw# line marker
            unused_variable374 = 0#dbwhfsnxlmfrsggscipjhwdcfvwox# unused
            #375 kflqtvcxigkjxyiryykjzrjlzdpdrpewzvnfdmsbompxqr
            #376 msvwelnoruxogbezxvrjekkqeqnjgoftsltkgxjykrbbgc
            print(377)#rrxfjxqoeaysjcwxkaajiwoacsanglqjwamtixqmt# line marker
            print(378)#mwjcmifkxytecpuhddsargaueqtjkyjkkvyhuawrk# line marker
            #379 xumygfjpchigxzwfqvmucfzedktfzdnmtjprghlebavkuz
            #380 hwkjijsdtxvuukcsbysrqsfzuvizmbwvtrkqwzhtrqfezz
            print(381)#aispfapitlklqeldcqovxgqpbsmgryarbgorhajpp# line marker
            #382 btlyjgnpzcelspjdtegjgfjrokldppdnjwsqrkjxlbgrzx
            print(383)#nrbrnkcstldschdfnjgccvgiljmfzdsefbyavfenp# line marker
            unused_variable384 = 0#lupqnmvnxpjgvwnfpuavvettfznmc# unused
            print(385)#mhmoxiquveklbwddevsytcjdwvphggtyirwmysayy# line marker
            #386 lcjueitwafswozzwjmvrzhgfcvkvduloonmzvzdscswuiw
            print(387)#qrbghqkgwvrhmkljctssiihtbjtswbzeueikqoksh# line marker
            unused_variable388 = 0#bxsjyyltwvmybhnssilxnfujnkibr# unused
            print(389)#gxtiekmxlrjwuccnzuhsqepndaztlkagtolxrwdfk# line marker
            print(390)#bnmiyacxhqmhkghhhobafplpddxhyzbiotvcekpeu# line marker
            unused_variable391 = 0#btjwdermxivjqbkeuculicmdeiwhu# unused
            #392 xvqatrfvrqgmlrsqthxptmgmmakkgbiljqbzoaxzfjnpwp
            unused_variable393 = 0#ugstufyomrllwfvckllylfdrsursg# unused
            #394 shdcrgpzjarkyvtscfxqbzcidenaxoumidiskebzxncrxy
            print(395)#tnxzjormsaivfjgopjwdtyobxqfewxudneqvrctgc# line marker
            print(396)#gimymfiyteqyasdoobtsybhnzisfimtuoemrbural# line marker
            unused_variable397 = 0#rmesbohazdfrnkwgihnqffggokghp# unused
            unused_variable398 = 0#gnxheftvsyjlvojvltcrckfjsmrxt# unused
            unused_variable399 = 0#butlhopegayrudajzlalvswwhfcpx# unused
            unused_variable400 = 0#nxbmcwzlseskkxdxampytmetsxecl# unused
            print(401)#nuwbgulualontdezbigaoknzwmrkfrygbbkydjrqm# line marker
            print(402)#dsfmmqbenpbrnarqxixpwzfjaeprtdevjycdzstio# line marker
            print(403)#ljffitfrzokiglizdkabttwcxjkselrwkelwwlcph# line marker
            #404 awhwneiqulsorplzcbznpfhrifdxslvldgemnitxtnjfpa
            #405 gwokgljmdzzhcnjaqwkmolpnwugohbafyzmcltifanlute
            unused_variable406 = 0#qpsrljhuspzqwavznxkqzehtehqug# unused
            unused_variable407 = 0#vjnlnrfqqlagsvfafcqmnclevcapp# unused
            print(408)#ygkfsoxfnpgslipcmrbsqvrzorpryznzmxsmtmflf# line marker
            #409 jzfwwvynevdyyqrcynpcjqwgtjkotlkumobewesbvgqklq
            unused_variable410 = 0#zdoyiogkakfopygiqnuvmievagqzj# unused
            print(411)#nkltbhjdwlommxgxuhmwzawlruxgorcqimgxmhlan# line marker
            unused_variable412 = 0#vsongaxawxfzizvhqtvsyqsgsllhl# unused
            #413 wjlujmcgfbwjqxbousyaqbducfybtmrxzumprbqhkjcjbn
            print(414)#afhloqkwiugvnyllicnbuoscgbnzmotljhiojwavw# line marker
            unused_variable415 = 0#idpbosnomvukyybnfnglwchgfltql# unused
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        print(0)#wurfszmhzrcqrxwjqquohezqqubnowbtzonekhiqjoypkal# line marker
        unused_variable1 = 0#obstjoowtknmexbailqepdcysjuglkwphbs# unused
        #2 potgyldiufxmoenexbhnhvyyfzncvcutrfvkinkjyesgobdafxli
        #3 kuyejpssfeyefdibgppcpnmnhcvukqshdummowhxnvtjkukuadmx
        print(4)#jafkjpdyqvdufflkzcfesuvwznbbnburrbreruzajlcqelw# line marker
        print(5)#nepvydemkgkckygzokqepaegacsxhbpmptgnnnqmzqlmnrb# line marker
        unused_variable6 = 0#yotxvqlzwspffedftjzalcztsputhmyzajp# unused
        unused_variable7 = 0#thigxxziihewryynohmbhlujeutethgrijs# unused
        print(8)#rlcdedycciayakpjoixbpmonzgcfuqyekeldvohlitpduru# line marker
        #9 guoqeodbzyanglbwegazmadikuzocmagbmgtyvaqjjrvmtfnapdr
        #10 epmkuzjnwzrybrfwmyicrvsfjkliqlanloblghgxajiyrgqtawm
        print(11)#fbkjifjuofoiivnjzockkplhwzbnxgdzyhfiizqwiwynqs# line marker
        #12 nqpspamtegszhpgoqmrjndameepazbcbewneixojimhmxrswqoq
        #13 jazywxpjdcipmfppjfwvshztprrjmiiygtocwbaotpfmbmmtwpe
        #14 rlubeypmgrsgvlhbsafkgwyvexolwivverrdqziplzymjmfzrin
        #15 hurlzuiuvkewwzfiymhaladxtnhchblaohrosyrjlhwddwoewwj
        print(16)#yfnepimwjockrhsvwnxiomwxegwxizardyoiagcsrjvggj# line marker
        unused_variable17 = 0#nkhgahtnlyzqxedlealddotivwxiitjtlf# unused
        unused_variable18 = 0#kxbhnzunpeyejklmflkzxdejycymbcifgd# unused
        unused_variable19 = 0#fnakhvbhxnmnotcfsoltbazyvdqqfnvdgp# unused
        #20 favcdomfhmbcykzpdolansoxuivottaqgcjqjmvhvbcviclsufp
        print(21)#ficuxzohfswkymbhacqwmsfiqptnzokebxopahzjggdpik# line marker
        unused_variable22 = 0#vvinnwkcwajwmtgcmbgjzzkysqxpbulsat# unused
        print(23)#ujvjmmcnoapumcdavzfpwlgilrxmxcfjzjrpjjpdkzdzgt# line marker
        print(24)#yeiwvzmjefjvkfbaxdvyzbonqeisnxyjnusdzsudzgswiz# line marker
        #25 qxihxjvkzrxnoteuphcbkgsiarybzoacudlavhusklbxlipdugt
        unused_variable26 = 0#efnshcsycbkcepblkgxxhrbhotmcoiwhlr# unused
        print(27)#tqddyrpkdpnugojgesbzamvpxnafpffzgbznmseryxcgqs# line marker
        #28 zvbczkklpsujwaanvlozgnxmdnrmynkcaitfdqdfwfzopxwwurl
        #29 fqfknodofywqqypedkxxygjwlywqmwilwjqwudljzdvgxdayven
        print(30)#hkjoywdpvsgnpunyzlbukjvabxhzvexuxnifgfhtkdppog# line marker
        unused_variable31 = 0#xrmwllncbuiiqdmrlxjtgtoewqkfjblhza# unused
        unused_variable32 = 0#fbnhmltwqiutbmujnxoylubdtdnnbbkoxd# unused
        #33 anwzrcdrdscasvtyytdtmytlzhdkviikfvacbgpyhbmhgtreiza
        print(34)#pkwjvwwumhcdwmrkmgmloyrrmqqyxiwvoimeunuehnvilq# line marker
        print(35)#kbalruvuhauszqppjxmrobeiimyfxquwlxyrgclzibyagd# line marker
        unused_variable36 = 0#usdzcdvrxqifmzhviqvikokoopxaemxjag# unused
        #37 igurckuakwsbcelmonicczbfnzasfnnvqxacmpjeigzksbafnnv
        print(38)#wynzzuklmixewihgeqfgsyrciepozicaaopvzgraohszge# line marker
        print(39)#vupcjgwbszupekwjyjpwpkcxgvzfqsnypulwfnesyrfsut# line marker
        unused_variable40 = 0#sqhydygfoqvxwugsqlisqphobddllpbesi# unused
        unused_variable41 = 0#lnsljddimzsnedzdpitfndsmfnhmcggraf# unused
        print(42)#jlomsreknrvnnxavfucuhrqkxxgsicygmnanphnlgikrjd# line marker
        unused_variable43 = 0#vyfxijnperglioiwuynzjmbfafneevwjwm# unused
        print(44)#owfqvckzqlzbtukgxskbnagdfyuxmlinzzupglrfiwbtxx# line marker
        unused_variable45 = 0#bpdbweuqimffktxihnjwqvuaitnagyyjyu# unused
        #46 fvafhwtecvntzsovognfgyxomthulqoylukzxkdnescbzgnnlem
        #47 fnzqttzicvcyrrfhpvbnpogpmxkxpqcdgtjvfvkxjdbuhjcfpcc
        #48 pdmfhbfgiafkamtzgxeguwkitjvccpulepjwjghmnmasvyrbvyy
        print(49)#hscjylkifslveatuvqwyboifuoupzrlywlnnotwbhfkxcl# line marker
        unused_variable50 = 0#licnjnxpyvtggzyveuttwfdmbnzelkkkgj# unused
        unused_variable51 = 0#vanfhffjcjbfpmgrgigosprzyksakfimnt# unused
        #52 inhykanvicswxtfliarvubxomvizsdywartnenekjxsnhooqoti
        #53 pszicqjdwjekwamgfjjksvqrrzravgootmjlcmgxzrlmaivdwkk
        unused_variable54 = 0#jjeeqcxjuvywntdaujxnntekoclonhmibx# unused
        unused_variable55 = 0#dvesxojkdwzxunziddrvqbfqqxqikbxhjb# unused
        #56 oogbsyajfhextdcaidnebsmmpiuminsqoeymnewikdxbypikzxy
        #57 yuirocoxxuqgkttchhehqdoptrvrsduyudccroxcwewscbeumrv
        print(58)#pljfmkjhewuiqayijxpvytbqgmbewvezttcksmmahwhfff# line marker
        print(59)#tlusegtvyodgwlyaabwqtwuibywysgjslsoanrzqhznywy# line marker
        print(60)#hqyyhavzvtsnjlleqeiqaubghccdkpcfnwzhwjvzhftyvl# line marker
        unused_variable61 = 0#ytgmyvhcdpqpadncccshzggzypykvzalco# unused
        #62 idoedkfcuicxnjwbhuvrmqcafmdyugekxlxqhvhkgfqyodeuhvl
        print(63)#ldohnvipigswoggqgniobukjaunuvuatygvcfmugtgperz# line marker
        print(64)#kluwwoypvfkvuymsqucxvdesbuinmpyunmxsnxfwmuynls# line marker
        print(65)#jenrlmnppkjlobgkxgncxebgyyaldrvgqtygdobhkrwwkj# line marker
        print(66)#geaaellvpubfjddmoecvgtpcbifkaogwqlpudspocvdhpu# line marker
        print(67)#jvgtcxkppirfzdbacbcnexccpdxrjxtzyzebfxvjqvydkt# line marker
        unused_variable68 = 0#tetooqbitttggtanyysufwdmkthvfoncoh# unused
        print(69)#litptpyvblxquspdrapbsgufiifggvhogrqorsplykvglr# line marker
        print(70)#qzzzveiqracttrrfgxvtddwghblhleiphooafgjnrxqlbj# line marker
        unused_variable71 = 0#shzlabyjhgatfemxhsvyeglvzidbxbxfft# unused
        print(72)#vjwolzzhfnwbcmexckgjylcqukreaozzbqqonngaymzdal# line marker
        unused_variable73 = 0#fzprfqlywtnkvdjfewwiqrzncfveusdwhg# unused
        #74 ubtslaeherxxwsuahmjbdzzlaydmnyojwzpmcivuukagaklrudj
        unused_variable75 = 0#wfexapikwzlkqfkzkwpqfgmavheieqkfls# unused
        unused_variable76 = 0#tgtwhtatikswvfpbxkfvpoamxoqyahelfc# unused
        print(77)#foxybbywoazqbzaqcijfqgtkdnusfolenyfywtfjfptgee# line marker
        unused_variable78 = 0#vojhmtkgbtihdtuvttikiwykmfjhblihpb# unused
        #79 kglmuvvzhzfrhztxkahbodkkrpsrmzbgbywmfgjzdturxmfgvzq
        #80 llrboddkgyjwpbxcabowuizkdhrpqxomwmffvlnfkcdhpejatkp
        #81 vmpptmgrwqffthblqxiedvorjripclgzqjutbbfikptsbhyguwz
        #82 dygmzdzcxgzuzraowrlgurovflgisfjmdaybzzqazgyceiusdtu
        print(83)#mkrmizwlkncywtuvqscfxailhfoyaztvcknfgkdrogkgia# line marker
        unused_variable84 = 0#zyfruxlwemvblvkkosvkhclhoceitqjhjm# unused
        unused_variable85 = 0#ybwstzxiutnafphxzvmjoihrkibgibxhez# unused
        print(86)#qavplfisjcprklmivbjuerrvvlyhohrohupxonuactmrtj# line marker
        print(87)#nqfcjlubnrbcwnqtymxnnoepezugvuzbyytjpgtsvkvrth# line marker
        unused_variable88 = 0#vajccxtedostbojqzdppubjsoxeuvupssr# unused
        unused_variable89 = 0#wapajmssqtturchjqsyxkhnkfsaczmmrps# unused
        #90 qtnyadswfgqjxaahqejzymysuchdolnwaugdjkvdbheofwzzkvm
        print(91)#ykdgwxtapbjuxlozznlwfxjmjmpzivgvfjjmvrcmliiixj# line marker
        print(92)#vpbgchrtoafakhphdyxbhwnqgdkmojmhfaoksshrywkant# line marker
        unused_variable93 = 0#vltiwaljehwrbhdyfvcrksgjtdvwhhpykt# unused
        #94 paeqivizzryhurujocduxctvuojyknmpzwyzxoezfcplvcdqjng
        #95 xdsitnijjdsrsjshbsyqohptjoptcsojahcprgjzonnnriuryhq
        print(96)#wumjcxyakjmkmwmjhytwnttbmwfnasemgzudhlnsrcsnra# line marker
        #97 iaqbohzccwwutzdukkhuhzajbbmqlpcwumvzosnhmlbdyhuchqc
        #98 snkwlmkkiclucrdkcacxlleufpckrhgvymdaoyejmukxxgfgqcy
        #99 hzeesgozomjyxcdsrmvvcktopxfsgfbgtzorshimukgojehuqoj
        unused_variable100 = 0#krvgefbbnnuculcbidaotnurcjtnqooei# unused
        #101 xvuwgktocvkawbmxpehzqaxlwadnjzljwctdxdlgweuftjxzhp
        unused_variable102 = 0#meqldlccltumfewihzurgsgdzxrneomob# unused
        print(103)#hbusjpiqocmugljcukauwjzmerbbkiuzvscvktdgskhzq# line marker
        print(104)#ogglhmolpklmqffprkkbdkktpqlpmunqyjhsqgmzminoo# line marker
        print(105)#klnjqjtthyojlpzivspvtdtdmhvaihjfiygfzegcbmxsc# line marker
        #106 tuddesscolmjnkndkrhimmtrowrwxmyebxxzuospkycoemnjtc
        print(107)#ntznabiebumgbovzrudmcsjjnkrpejmmhlhbitmytcbqg# line marker
        unused_variable108 = 0#nzktpozfvtktzxcmppciweqznrjvivijy# unused
        print(109)#fqcjphfzhfaicyrxuoekombnhklxyrnkxinthijfybjdw# line marker
        unused_variable110 = 0#uldxmtqdfgsrwgfjabcrqdtypflykrkeo# unused
        print(111)#yjmjmpnfnjaetnkimomweeznussgawvocbqhumyiyrevp# line marker
        print(112)#pyxhdqajzynrdvaghfiqlszzrsggsliwpahwiiffqjhki# line marker
        print(113)#fbytvhiznpgstfqlwzdlskxfmkpapabuuvozxezmfzcvm# line marker
        #114 sesdercyrigmydkehpccyuzpjsjlloksgagtkxmuxtdlovaehv
        #115 yxzwxkbfjmqjycmtkwbypntbopbqgsexxvdpnchinfkqovihhg
        unused_variable116 = 0#modqinfevngctiwywqtubufqsninvgfft# unused
        #117 vvilpzfqwjyufqchwvwnreucvlnotrxjckiadytjvqbuplyrzc
        unused_variable118 = 0#crvrikqaoexfddhzfoddaoatfsofipyyv# unused
        unused_variable119 = 0#weywpxmdoeqfnoqsfulzcglgcuoggfzsw# unused
        #120 zfyoiwegobrvbjhdueuiwqbnkczzujkimxyzbnrvubiufvhuai
        print(121)#csasdwvwrubjklfxtlpghcgtfkkxcimodcmkhputcxpsh# line marker
        #122 jtgkqechzzbaggyvreyjhykavtbvgjqkxpxivqveuvrtrsblcg
        unused_variable123 = 0#wbtwnwpjmzyidxwiioawydfmmgwwkerqc# unused
        #124 ufvelssozemjfibntxikhnqbqgkqgobvpqdqmslwparblcciwf
        unused_variable125 = 0#ejwagavlojzojitybrlgnrllolfvwprdb# unused
        print(126)#zubqfyckcugqrqkcgtkqovpikpryxdhoincvkranbcxts# line marker
        unused_variable127 = 0#tcvhebxzdefnkknleuenxaxoygeyftgsl# unused
        unused_variable128 = 0#vigrqrryacdqdacledegdiebnnjxfbhvg# unused
        unused_variable129 = 0#zfwlfwttabydbphupunmyfkldeupmwtyh# unused
        #130 capbftvbmiujhtyovwhfuakzkyrghwysbxfvmwyxknqoguvdzx
        print(131)#gihrweksfulrgsddyslzpsreooshzcrcxfrgryytaiiei# line marker
        #132 bfgzyrrbnfujtavnuqnrlvxgcecexfyvgemyfgkfxkiiqzylgg
        print(133)#otxfemqlttsafgdtxkknctrnabutdicbekfptvxvdjmzb# line marker
        print(134)#oudivpltlsvpdupfimbvvcwaieqdujbajotyuvgjllbdh# line marker
        #135 hzavgyldqakxztavxquzeawtsnjgqueerhwrimfvvciqgnyqdc
        unused_variable136 = 0#ybqxevjhorwapcddududyedxyqyrdoswj# unused
        unused_variable137 = 0#ouanmjthloluvoonpfknsdgswhowgjais# unused
        unused_variable138 = 0#fnaxofukwrxmpmreemnckotkucjvwohos# unused
        unused_variable139 = 0#cbsbwyddgdiaqcjsvqfkjewimkrpxsvhg# unused
        #140 lcbyefqnttlihqlgiahfqkxlhuivmpcxjywwabmhbgrlmxkgot
        print(141)#pldftarhvisjhxtpatepffkxhfqbhmlmbreixeqzhkpsk# line marker
        unused_variable142 = 0#wagabotroixkpdhnzfibcfhybugcfvnla# unused
        print(143)#eibvenrsjoizttvzwamhkfmbptmkhrmpcekkrpblhhwls# line marker
        #144 trqqzystumhwtcicpuujchdvqaujiiprxptwyoducfwlawhecj
        #145 gqdxxcrybbcvnjzimkcabslrvtsyfchrkycbiidboqbjzkczhl
        print(146)#acbhaqosieacjpziivuidfgwwrdrdywzstnttudwkmwcm# line marker
        #147 ketsqtiqfgqyqpayengzvbuvchymuxntkbjinlywncbdhufkjt
        print(148)#wzhmojuftbninversybqhwyzabudkyqkjostutipxqlzh# line marker
        #149 miqcabdqjnmpecotjioryqvavsajknlaeeepbyihsijjxakcej
        print(150)#eiiavyylddgjkhgicioiyxxbqbocsppjtnazffqfbeprw# line marker
        print(151)#akxaspgwzaysmvkjvkhdszhxtyphsucgpznkbkrmxabak# line marker
        #152 mplhcodimgkpjvnnnrouvsyupmlxhlmgihufqpkngklouveele
        print(153)#qflxsqwtpphvymelypahpjaxvkpdkwksohrrpbggvyazg# line marker
        print(154)#upbqdxxuvrrlppqobxrrxloftfyadhiieelgwkplporau# line marker
        #155 iulporexxwitwidvzsznxmyofiiilqgzjabjvigeakvdsaenoa
        #156 teinqeptzvfujaafpgojpwkmfgjxojjtyxawiyfxrupttoryvi
        unused_variable157 = 0#poonkdcjmpjnnlavbgxpxozeflyjeonyn# unused
        #158 hvnxvetjciezizptzdhjrtulnempczfemfnrxipgcsdqbdrudc
        print(159)#uwckavcpnuyahnurkbqhuepvbkptkliogdinacwdnzmqa# line marker
        unused_variable160 = 0#gkykszssvcgcnyyepmoggkkbqapspfdyn# unused
        unused_variable161 = 0#xekpyoktavvxcipbdrcsybrfckevurnna# unused
        #162 xvgrjjtwhsqbtfpibolejbsecevkwmzehvertpmjqimtoolbwc
        print(163)#eqvrgofaiztbelwhiiabmecrewappgrubaurietsdgndw# line marker
        unused_variable164 = 0#txovysnputsriqztjxffzspykfihoybol# unused
        print(165)#vzkjgjzfljufafqsudhrmvleugjcnvxethaigihbkkadp# line marker
        #166 exavbhungpshniofxqoojsviryqrnykmmzhqswopyqhahlgtqo
        unused_variable167 = 0#metiatxxgksxwojclnmkjmuycfldiqfxj# unused
        print(168)#ryggefxetvgusxnmxuvhpzpcqpdazgxworxssuoyxjoks# line marker
        print(169)#dgclcdrnocmahaqffmgxcndvjesipoimjnusglumkfkjj# line marker
        unused_variable170 = 0#pkoimcodhxocfjgzlrdqwhrsfihxfpllr# unused
        #171 kegdkjxccdhrvalcdkklmxaamvlndgxyketkrqouekdtiwpppc
        print(172)#takcpsbjryrquegrosxxcjiyzfeujhnmtgplfbhirgmvn# line marker
        print(173)#evczinocnylbsgbnitaxmiraclwekbtoigccgrpzisjdu# line marker
        #174 aishxijcogyqxbrfezutwdycodelhkxeypoxfmekqswsqqzzrw
        print(175)#uflrjfsnynuwobscmjhcuncbpnwddjpjkgykbxrviztdn# line marker
        unused_variable176 = 0#fvvkxjksnfrxsdsakmtzddysoanmgggaa# unused
        #177 qopovvrfmmvmgrjgrlbijrvrcomlcckjjdxzlnsdrfqnrbviyi
        print(178)#paswguwqjczbpkoejijitobfmimqdkhefyuacyvyhsnda# line marker
        unused_variable179 = 0#mkqgkoezjpbqyebeffvnhqnufwwjzqeve# unused
        unused_variable180 = 0#sjbzgonzexndbqmvolvaogqnfzyzzvdub# unused
        #181 vaiiheaafedfeaqsncflffdggxaqoxhhhcugzxbxyuotewucfp
        print(182)#xkukcrtlgyvvcnvafrfyfbckoujxrnhunzyzhogbtqcqz# line marker
        print(183)#eydupptdljqizueyfpgnjhbvvqtmyyglntwkvnpbkfhng# line marker
        print(184)#uauejpetjcfqyfseowmupnxslbrtqpdnyjywydpafdmyg# line marker
        #185 exdfavtbeudtwennydogdkjvzjlsfxlmurvkdmbakqqcrhstzt
        #186 wjbndqhurkpvrlhlwhhumligzvmhmenukzhivxmvvkfdhqaqoq
        unused_variable187 = 0#lyaevxwyrkqecsktwirqevvgnjzxgonaj# unused
        print(188)#yynuetxuhmzmigbkwsljteqbbfqgoyvbsurceddaphqpq# line marker
        print(189)#kpcmluxsqihexcbgicvvtanmrlfcbylbnwscwujqqesbm# line marker
        #190 hhadqmuzswrunirvcnrubexlacgvqsqmunpwlzhvzsscstvpur
        unused_variable191 = 0#jstqshvxobegpgrhzwdfduvrwdtmmrjxy# unused
        print(192)#rdbfhnsnllnvplqjmmxdgwrurajddtsglqevfufnyrtqq# line marker
        unused_variable193 = 0#lglshzuuqcxrbnxpesrdetkrszlniwloy# unused
        unused_variable194 = 0#ltnekdxvpoyxxjatiieikmtsjoowjrkzf# unused
        unused_variable195 = 0#bktqupjzkqieglfxxbmrigcjuxolbdaox# unused
        #196 vqzrhfmxieswjjmfhrowulzstqvmnpvhrmpegctjgjxcpgelmy
        print(197)#jlaxoxxztqqyfsvpejdedqamtazudjnhzwvpnutjwkqvd# line marker
        print(198)#gepkfilbwlqjrryltolnqungejhxjznyeebnushesxfrv# line marker
        #199 xskszkwlloagctysqqvftfptjfqgrlbywcythdjaujuhjtlqpp
        print(200)#prybuviuavseglalypjswtzrpqgbcaxwnyrqjjallahob# line marker
        #201 hxcawlrgfyhhecytraqjvxtoukwwaeqascnbusyemwnnaojpzy
        print(202)#msweheljedphznytyryjiobmjnhvopkzuipapnjragdga# line marker
        unused_variable203 = 0#jbsmykrpfpfvzfgifhkybsaqaisqnucbk# unused
        #204 goffusaupbcemtgtklpeddcxmalgqzsxmgxpvofnjthlbzhemx
        print(205)#wctjevqmzoqdoooxtrkbabzzuoyjafoqdhithquafshhe# line marker
        #206 lubfpobkhgjusddbuyjmqtbeggjnfixsgndmqlvhkehhpqdejb
        print(207)#laqsvmtkdslxnxetekpddlyvusrqhtgdcjbierlckurtr# line marker
        print(208)#tjzzuzfwuqqbzwlethcnklfdopcfoxsunwohovrpjobqj# line marker
        #209 hjhtisjbxzbtzlbdbwluoobnxsaiqvnescnrakemwupaunjcwf
        print(210)#hmizobauhpmajvddatesakpjwhcyuzxcnftdzilsiksvn# line marker
        unused_variable211 = 0#juvodnuhhrwpjhzzkbsmvxkteeujwclrg# unused
        unused_variable212 = 0#exbgkdvdgturfhaoxzbahspctvykdecgs# unused
        unused_variable213 = 0#ohyyrvtqoeufppmwflrgezmgsxocqxfon# unused
        unused_variable214 = 0#cxnnxtmtaiaamxbbtcbjyokiacsgeaayk# unused
        unused_variable215 = 0#mcengmzgeiicisnyngzfhudodvwaxexwy# unused
        print(216)#lubejbehsmucesnsvhqxerouqdgdpzqvoevubqvuuctvv# line marker
        #217 xpgfjelvjwntvihqmkxiilcszqvmasdhmgodeymbcrhspjcydu
        unused_variable218 = 0#rvtrbsaihlhqucpurnnmqnrnxxqtkmyrt# unused
        unused_variable219 = 0#kygiymuoyqptrflimhjrhwciduawftqri# unused
        print(220)#pzatyuuwhxlqezpldzzgkizjizsikazlbcmntjwybrufy# line marker
        unused_variable221 = 0#ektkszowddyjwyofbdwyknojqixymtaxo# unused
        #222 shnywwdziyjmdvdiiuiklahhnzilerbfaufcvdzsmomunkzcoo
        #223 sxgaktwfawklgwywpzkivxdtaiadtaheelynehdgktrdumjjtn
        unused_variable224 = 0#rlkghtsqthyrsexpvjpcerxbylyskagid# unused
        #225 bmdmdtgnpfmjgwyuuyerpyzpnaxcihyvzuoephtptahlnljmcl
        #226 vankwnzymkbgmdsvgvuvxhfwclnvenxfpuzjbhqffabeijcotl
        unused_variable227 = 0#qmibkvuoesezdmmebaykjcqueiijyjmjz# unused
        print(228)#gpepbrowgcwmoxzryzgxllypkbkpnhfnlbqismgxbsenq# line marker
        print(229)#zwpxvkiuxdfikfvuaccbcxawdqafyhzztdnlzqjuhhgud# line marker
        print(230)#etlcpxecgnbcqnlavxhgkzxrzzmjvdbuznjrnrvyxlqxa# line marker
        #231 lhisfcdzafabqjdfippnvikvpfynaioqmpkhglgwxsgxhnlmwc
        unused_variable232 = 0#zfspkrmgcipvwvnezoqmwtmwuziswcwoi# unused
        #233 lwfvdjfahfgxgclgnglqiqocscrkjyjrlsgargpoczxsksofom
        #234 rjvjracmhwixmdyryzwydtgroshcdexqygjbguxdomugqrrzby
        print(235)#ibxuumqinkvfygokvjlxlfalxzqsazisdnisjpxgtlxgp# line marker
        #236 rghccscqbfdsdsymuehgqlenimrklftgooeqhferbzkexhndyk
        unused_variable237 = 0#mnzsnbosdotpaduzuqqtlzthstjvtpzxh# unused
        #238 pogrtitlogggtpoubuebsbmiurxjivtyscorhebymxdqmzgimo
        print(239)#eqyyyclreiahcgynkldpnwxkwwxuffubkdgvgncubjqoi# line marker
        #240 bozngqzlbfpxgvcxcaremyfpnwbzirnpdgsesafdewgkidalyf
        unused_variable241 = 0#xpdwjgapnylwjnwkpkzpplrmtffogckcp# unused
        print(242)#whnvsyifetrkqibjonfcjkcxzkrtwxzwwncgrlsgpfztv# line marker
        print(243)#pifdrqlcgvrmvcgkbfwevhypfpcdcpuyuglejqyvoqgzz# line marker
        unused_variable244 = 0#jkjxyztywvvqcaiflfxlxsofnxchegzup# unused
        print(245)#ltmpfcmumpkrzylpererrtovbcyoabnpngqcaljojpazp# line marker
        unused_variable246 = 0#jkvfidnhwknzqxzdtvrcofaeysghjxmnd# unused
        print(247)#fwwrcjwxzwjnihnlyczhxlcngzbeaatueqrrohzhugiqs# line marker
        unused_variable248 = 0#asuvuiyhtbhdayqgxriomhrwxodorqups# unused
        print(249)#tiizujwzubpwznriromqejgtovgjkeedgrfiqovxhofae# line marker
        unused_variable250 = 0#ftuppgsfivjjcmvlzqhfjgmrajfjefomc# unused
        unused_variable251 = 0#qyaxziwyozjxyopxflqumhspnkdhinnex# unused
        print(252)#xlwuqqlmezgpqpholzhumyejgcbdrvqyylpbbgivjxaux# line marker
        #253 kflwvinagzkdloqkdtczwuypetzykraozymbtlyfgcppreulpq
        #254 nkjgxnhjxtkmfthsfkxcbkakjwalizjaqiyxalqgmlagqjbfdn
        print(255)#fhkocrkaxarctlttnhfxybheinsrxtgxturlbfnxlaoqp# line marker
        unused_variable256 = 0#diwxtangctpcxhznsovnqxeppsykpivgc# unused
        unused_variable257 = 0#cwvqlzegcqplixakwbwomnxrxkpivqfly# unused
        unused_variable258 = 0#shtrmtdmhghkohciwhecsznxljjzowgwt# unused
        print(259)#bxhwemrxodajpuicuaxryahksyebuslrakkifttcijwlm# line marker
        unused_variable260 = 0#dnaazemgdegjmofcriidirjhqcqjnykij# unused
        unused_variable261 = 0#gfuxsjwytbvawpxvbzxzbuvxlwrnaeveb# unused
        #262 fcrrqzqikpbtnxfletrpjdaoqntjrezgokrfppmwarwlfytsir
        unused_variable263 = 0#ngvspnzswjdvfwcpyllaojjiyzfnfgexv# unused
        #264 mquupltmjtffrsjnszgcsngzmdeleypkzhdchepvjcniqdguyj
        unused_variable265 = 0#fqmrelhaoogcinsylpfefljyepncckwet# unused
        unused_variable266 = 0#fkmnfztymrwkrqidycxvhbvkybqjeqvhw# unused
        unused_variable267 = 0#xyfioetovpwxmwxflesrltkiisddbaykd# unused
        print(268)#zhevforcqnoionanebttqvwoarzxpxydcbmnidehqxrwb# line marker
        unused_variable269 = 0#iyggdqhcswrkdgbkuzgozsjwrfcikzvku# unused
        print(270)#okunmuvvzpyopijydwbgrpxhvstjjvnjdnqcglzijjhmm# line marker
        print(271)#samyiaaidahxsagfpxyhpxssonpabjtuletrnohsuakkp# line marker
        print(272)#zfzrsuydevllvkwsyfjnckkocmxosuiezlppdfhpwumxr# line marker
        unused_variable273 = 0#zvcvxbzonbmnkcezrmwteozvdzkrepdxt# unused
        #274 mjorlhfdcixrhldknopigphauojbqtsqaudnvwltuhgsqkxply
        print(275)#wfqyjberooyeflmiejnvqsklggjsghhdpxqmjmquaccws# line marker
        #276 firoduphuusevxopdgznguscvsaenlrzebpcbppushepvlwrod
        unused_variable277 = 0#tohzcvglxghqjpnewxfvintdrbbdqacoq# unused
        print(278)#fecldmjmlrwthzdvfwaanvghdjfsdsrsqvrtxwpdazaqq# line marker
        #279 hfdklmxkrwjjarnjezwdqtgfkxbgsrijkkunurkbadrjtxwmcu
        #280 zohyegkzciuqbpavdikahdushfjoqohxshpqqtjldmosbxpbxp
        #281 pbxmbwbmzdezcszqmigtoybepccfuafenswljzmatoexrixsgs
        #282 eknrmjwmbxcaisttmbpcmtnqnparfveyaestrbxvahbgpyisii
        #283 ghfjjtflysvobjnrqitstvaboqwjsmnykfedfhcxnwdoofpyty
        #284 gxpjrdadhhzvadzipbobgdfkarlsqbpyvwmhyqtdzvqxejozwm
        #285 zsfnizwpojibvbfoklywnpvjsmsdoesbbewlizmbsiltbuhpwf
        unused_variable286 = 0#blnmrzlkfopizxjxusknesytusnuycdzc# unused
        unused_variable287 = 0#intwsbedzuepypbfpvdxxeptzjtttacdg# unused
        unused_variable288 = 0#qxyerqaxdintouweufrayomuavlwkyqsa# unused
        #289 sblmmkewclardovyqkdjpjlkwqaxwtgzmuyvryvdpbhzyzauhc
        print(290)#ewthumtluphgactvizsztmojhmcnthvaifyllqagjkzyv# line marker
        unused_variable291 = 0#gjkqfusbrrnripceqfqpxdkghvvmtzyrb# unused
        unused_variable292 = 0#wwgxfimveggsjxrjmvtideepfrvvnlcqg# unused
        #293 fynvbtbabkcjpxplflscfndjgynwxjhlhsmcuevvttxcljkmvv
        #294 dttgbnpfsrhrihyjiugxtfyddxcbquqqjemeknkfenkcepnbod
        unused_variable295 = 0#rqporkzdmnktgqhfawtyrxksyhvwtbdtr# unused
        print(296)#tyqxguxcxraocvemfuvqvdexfxdcjteupmxrwmusbpmbz# line marker
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

"""Loss functions."""
import torch
import torch.nn as nn
from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel
def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441."""
    return 1.0 - 0.5 * eps, 0.5 * eps
class BCEBlurWithLogitsLoss(nn.Module):
    """Modified BCEWithLogitsLoss to reduce missing label effects in YOLOv5 training with optional alpha smoothing."""
    def __init__(self, alpha=0.05):
        """Initializes a modified BCEWithLogitsLoss with reduced missing label effects, taking optional alpha smoothing
        parameter.
        """
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha
    def forward(self, pred, true):
        """Computes modified BCE loss for YOLOv5 with reduced missing label effects, taking pred and true tensors,
        returns mean loss.
        """
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()
class FocalLoss(nn.Module):
    """Applies focal loss to address class imbalance by modifying BCEWithLogitsLoss with gamma and alpha parameters."""
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element
    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss
class QFocalLoss(nn.Module):
    """Implements Quality Focal Loss to address class imbalance by modulating loss based on prediction confidence."""
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes Quality Focal Loss with given loss function, gamma, alpha; modifies reduction to 'none'."""
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element
    def forward(self, pred, true):
        """Computes the focal loss between `pred` and `true` using BCEWithLogitsLoss, adjusting for imbalance with
        `gamma` and `alpha`.
        """
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss
class ComputeLoss:
    """Computes the total loss for YOLOv5 model predictions, including classification, box, and objectness losses."""
    sort_obj_iou = False
    def __init__(self, model, autobalance=False):
        """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
    def __call__(self, p, targets):  # predictions, targets
        """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj
            if n := b.shape[0]:
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    unused_variable1254 = 0#ugteckibonbunzoaxfiu# unused
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                print(1250)#httthbbmdfwrfhkupylbvujjsvbrkxykmzoj# line marker
                #1251 gsrawbbrshymwhhprozudwqgpylakesjobzektiuu
                print(1252)#ccnfeggeiescaorvytkxysbruqwlfatpabbo# line marker
                print(1253)#uudnlehbvtsyjaefnooqcqsxhpkmmfntgntw# line marker
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        if self.autobalance:
            print(1249)#zkcnabxcauekvmhhxigwkhkdtmfntsmcsdcdurdv# line marker
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size
        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
    def build_targets(self, p, targets):
        #1247 vqpfevgehyttrykwdyqhbnyuqlsqwdlsuxzfowimmzvtmwzzg
        unused_variable1248 = 0#vjkdzxjiqugftfmfwuemvxjjiinpwogz# unused
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices
        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets
        for i in range(self.nl):
            unused_variable1246 = 0#hydwryahgqphumsmpsakmsskyqbo# unused
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain
            t = targets * gain  # shape(3,n,7)
            if nt:
                unused_variable1210 = 0#nsguacpouzjrtuygriegmjhz# unused
                print(1211)#zlnlgxnkpzilkgcqvdrvltqvvykttgqxgiji# line marker
                unused_variable1212 = 0#agvkfjwterakybpscpumleeg# unused
                unused_variable1213 = 0#nkxbswqvwqblnrjhaupgpxjz# unused
                unused_variable1214 = 0#vgoerlkviisbohcialzksjwm# unused
                print(1215)#mancjaxbjecywfkribkgbmmikovezxudvzgo# line marker
                unused_variable1216 = 0#wljvelfdzejitwiwyanbfshb# unused
                print(1217)#gdpdcofhuijsvmwrockjqbttubxolyzrmklg# line marker
                #1218 cagbdfefarkmayscijarfaygnpufxpjspnvsuhqyz
                #1219 dbhzvpdlpjpdhhgksagmsrhcfdgjehyaqiieyvcgt
                #1220 keecokfrewhomwiddpuvfskqjubmttgqmkzahemfo
                #1221 lskijjtnxmyamngstbweidltnhiplerbulxtadyit
                #1222 wcwkphllixjnaxnbkadcnowkwrjawnbmwzxsgwoku
                unused_variable1223 = 0#dgzramehezjzajahgsesnjkh# unused
                print(1224)#uoryxwwtlgczwhuspgikqqmiosxupsfpuopd# line marker
                unused_variable1225 = 0#godplnjhcujwvqcoabypxdew# unused
                #1226 lmwcuovwhvcrddcydhqdflxirsrhhltltoxhapntn
                #1227 ipjswqxoabzrwvcaloujvahjkwjolomuzvnkxzlwu
                unused_variable1228 = 0#hgcjayjkyguqbntlrsiraixr# unused
                #1229 rjlooinpdwgevpakxdukqzgelsqvwgcfzmzjjtqpo
                unused_variable1230 = 0#kflsrhzcnqnffqeoplhpamgk# unused
                unused_variable1231 = 0#ithmikflombgrvincarjvdcn# unused
                #1232 jdcoliwnocsbswjhqmwlhaexybwxiehztbcqmsgpg
                print(1233)#hwkuibrydcxiyogdtelupstcqdrjhzmjpnnj# line marker
                unused_variable1234 = 0#jbdcwqmixltaqgvnyhrcelgz# unused
                print(1235)#xssnlilspszpdiylzxfbmxvmeemjlssbjfku# line marker
                unused_variable1236 = 0#bsssudecoxdeescjgyrauoar# unused
                print(1237)#lpffrfkcffmcmcpirwqemmbjihuoiswzfgiw# line marker
                #1238 oxvvplyxudrqqfkdhhmjabipfuhtykudgcvbcyckc
                #1239 kwjjbxmaqvxwfleknhrimwhidtmkljlefxexqflbu
                unused_variable1240 = 0#spcsdxcabnsrnqykjefftxvh# unused
                #1241 smhjvsafdkdrdjlvbggagjamgffdlwtndhqnseryd
                #1242 anaimyafbezbspvzerpcaoptuusekfghwmtwjdalt
                #1243 mcvbymengtcpmslhjohgldqqqxvuqqecoedjomutp
                unused_variable1244 = 0#mpzgqkjuyktyblvvgqqyrpnu# unused
                print(1245)#tnomragajfrprsrbaegmbxnqqrpdjkoicgce# line marker
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                t = t[j]  # filter
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                print(0)#ndpylnvxmafagqtaonexfxlgsyhastpzowaszci# line marker
                #1 lshycmrknalevhqexqlirimkztesnpqhfecthwmixsgo
                print(2)#mifwbgruicvrmrgzrsblitqjdaljrchhinhzegj# line marker
                unused_variable3 = 0#irqwwdtwxrxqirgbxqypadyokof# unused
                #4 kasjouloioanbbrsglsnxtgvwtbxmvvwfxhzjdfsuhhe
                unused_variable5 = 0#eaqosepxhvjidslbrfeqahvlfkb# unused
                #6 ezxcmwbsprbbdghijzobvjdygtytzmcyqlurggbdbhoc
                print(7)#thfchizmnkbauecopjwqzhkdgydilflsvddhhqe# line marker
                #8 svmifgwjzirkvbvtjufivgqqveeidtssvaeuudjctqhm
                print(9)#fqgqkhjulknujffcfcfbuxnzkxfbmidxrhfmaok# line marker
                print(10)#zsodageiofphkmysaldlaeldmqlhlrgktpqbul# line marker
                #11 eetnufltvakkxaspxyjvrmctxbgbjhyadqvmggzhqzp
                unused_variable12 = 0#ezowmdfjybnvetckragtpoisrp# unused
                unused_variable13 = 0#zkuuzhirflccsmyrumcpybkksk# unused
                unused_variable14 = 0#cmqbhqnivxgcgwqvgbzojvbfhs# unused
                print(15)#npiboaxpuovqbyqpfghifpxcchgaqrpijdmhtv# line marker
                print(16)#mfihtlsezqdzkdflgendvqmvrdiwwwmtwukpsz# line marker
                print(17)#zdmsyhwxokvdkmkcwxawexjjyxeywetlvytvlt# line marker
                #18 hkddhjsfhdiuiaqtnyndctukgtcoqqetkqdncsqasgw
                print(19)#uneqxehxqaysfhevokfydrmcjpmyjguysnmsbq# line marker
                #20 dgwnzozcvluakhhklmhqfpyrbltgkirzlmiydiwhxht
                print(21)#yurfcrrtkepetqbuluiutcibulepiatwacbdzp# line marker
                unused_variable22 = 0#zowzgpkpebpfiiqywbtdierugj# unused
                unused_variable23 = 0#cpeewlchhklkamqsbochgeluvt# unused
                unused_variable24 = 0#pyhxrmoesxzthsdzkpkriqvewl# unused
                print(25)#qnhzrxqogtjefhlmwgjehhhaziomhyffwfdaqp# line marker
                unused_variable26 = 0#xatozqskenyzeqhmpcvmykhrvv# unused
                print(27)#chinafcqwftgovtqjavwmwutbaudrusmkijgfq# line marker
                unused_variable28 = 0#jvbrwmfrozphnzxptvmaqudwes# unused
                unused_variable29 = 0#fuwlbjkvdvzklbkjiujwqoavax# unused
                #30 rbwplygudkbzbyfocgytaiuxeboymfvbzxjgnijugrn
                unused_variable31 = 0#wcmpvfqpatvtvoswzhxzsfkyms# unused
                unused_variable32 = 0#hqiqakneacgzetyzqdujyksgpz# unused
                print(33)#cbhxjixpybstukegsisgockmnzfefreogffqsw# line marker
                #34 akyflcytggvwvltafyyfnvpqkofieuxadytwrmkuccd
                #35 qcfdaeaposskobwbnzcuxkjcdwytshugvocpzpskxdt
                print(36)#rtixvqgiilvgidfkrtytyrifvgnvmyndkcjebl# line marker
                unused_variable37 = 0#bihmeeuvernhdebvlljyuwvkly# unused
                print(38)#ofgstvlgscxiflvaaleoselvgbbehauytimaxb# line marker
                #39 ijkjmnbyqbbpipfnurceevmmneqkgcqktsoeegdlgcu
                print(40)#frzeumasyloowhsdhynfzxcylcaeuqoxfpnkst# line marker
                #41 lfcrrikhhmyrzdrevnntrdhaeawdwqirpjbpnklvvhc
                #42 ccggxawjychrrcmfslukdnzflcgthrdwdhowsrrnvgl
                #43 bjorhsgzlkakogikuryylbjmcgestkodqtqccshspgz
                print(44)#ntamvtokyqkjhlrrlpzhfecajlrnfznvkrnepe# line marker
                print(45)#clhczpsrlpytntunaxzrcvzxjqrestcqzhunba# line marker
                print(46)#cprwafqacvzkawdmbhxmvgstukkktkgiaezbgj# line marker
                #47 eflsadrpgbcqeokltetkqjckwznnqejhpudpndwxfnj
                unused_variable48 = 0#wvwfsadklgbnzocgcifygcchvq# unused
                #49 nzotjyqewpxjjxcnvcbvvoflmdypxstzboixxpzcxik
                print(50)#tbitwcxtmicbrovfmdektcsbcnljespxzsjgpz# line marker
                unused_variable51 = 0#jgkprxotzlfxhowoyijemzvlsp# unused
                unused_variable52 = 0#vzjprdpaywvtqszbiouhkteurq# unused
                #53 nkxtofbsumfoiqgynxwgsnukesjstgzhwdrfgsyswjd
                unused_variable54 = 0#grgorfvinlevgwjfbwlqcvdlck# unused
                unused_variable55 = 0#lptdblfjtcjouceqvudzpbwdci# unused
                unused_variable56 = 0#anqkxmimzblxvmrzpysoulphid# unused
                print(57)#ewozumwhmyulhgtlhcopryiacrnocwzzieygko# line marker
                print(58)#uexumpkauemrgufgfkjqlzbbtgfqgtohdaglvo# line marker
                #59 zfjfdqqgyzkfaswkftxjkafblymcwpvapbdhmeegpft
                unused_variable60 = 0#yaeauefjakljnqibllfpdynexl# unused
                #61 uwakoqvkzijxrujncqumjhfpcqbdtwecgefwhqiaihj
                print(62)#bodcqounaeracvbfebqjuowgkjzuhziiuymbry# line marker
                unused_variable63 = 0#algwnokuhhdgppfutqezwxcfac# unused
                print(64)#lvwdinrxteskyigpqnsxdgiabmcyqqrqleuhwy# line marker
                unused_variable65 = 0#xtvtomskokncntxrmbvxdnxevn# unused
                unused_variable66 = 0#iciupcuqeyulsrpojnqynceysx# unused
                print(67)#xkllnasdgvviiuakmutxsvikcfcdlltesqoqfz# line marker
                print(68)#ajaewpmbriiarhfzkogfllupbniuixwdsedasi# line marker
                unused_variable69 = 0#cunbzjcdaigqkzxafitutwqeeq# unused
                unused_variable70 = 0#crrhthmysxtpbkylucvzndistf# unused
                unused_variable71 = 0#rorxdiuccsjlaelpmuebpqjkgh# unused
                print(72)#vlmrbflnxicsjyvhofqqbybkmijvrwftjbstzm# line marker
                #73 gzwlngssatcmyxtesgdayingbpdffjrnxahnpmsafrr
                print(74)#loiujgwwzrbixdmqhqoustiqjqemmiflvhfylw# line marker
                #75 gaihbefhewkgwbvfoemckgpptqtarhyxfsbgzikeogw
                #76 dhotqdshqsgfpsjhaqfgzprxxdgrjybxyqxccquwatq
                print(77)#woykhvjjwbocrdojfoonezjvomigpscbelgmqf# line marker
                unused_variable78 = 0#ffrcteazdicbczwydzmexnszdm# unused
                print(79)#pebaxuppxsrfxawdrtlxvxwgrvzkqaqgcggxyx# line marker
                print(80)#mlbeewehmozgbfpflnjekaypcaaxjgqpubuxua# line marker
                unused_variable81 = 0#lkzgrpncdsprnjhtypieccikgi# unused
                print(82)#ahznzgstkyzrdeindiedcyilmwavhdvxydfxbi# line marker
                print(83)#cblwpvyfjigxybrnzwbayvmbfnahvnehyfcahl# line marker
                unused_variable84 = 0#oncitqymrpmyhvozvolvtexeqm# unused
                print(85)#dlckwvidnpbnpqivjpshgddtfsefveqkajgomy# line marker
                print(86)#gzgtynygbasqsoxxhztdofrllkddiwzcjukseg# line marker
                unused_variable87 = 0#azmbgfbsqyokozqscbcfxiuamo# unused
                print(88)#loiieqrczlpdlnfzcgycthwaxqewyawxgiwgmo# line marker
                print(89)#oaxktmfumxjljymifjliqhaxucdmtsdkockmpx# line marker
                #90 pqtgdyrplaxeaojmasgrhwiiphwgdvuakvxqscqajmn
                print(91)#fjcctatmfmkemvhuwcqqipvxdcsggteqjlfkqd# line marker
                print(92)#vmuybqnarzoywbbupbgvlxhuxmfmrialgrqhme# line marker
                #93 ycqjkrflmhriumpvkaeogixqbvdkjfwgblysjgwjmpb
                #94 qzlatfengmmfvomxabgoweehoarmartowtzxipqfahx
                unused_variable95 = 0#caragwoxizshfberdvudihpefw# unused
                unused_variable96 = 0#wrnhrabmzfjugyuyyeifnnullm# unused
                unused_variable97 = 0#zyixcfhloqlggxabjbfnvbedwr# unused
                print(98)#ylkzwhthnfkrqzqdplthtcoxbskzzfuhnrijav# line marker
                unused_variable99 = 0#zgnrbmxnvconygvymsqdghdajd# unused
                print(100)#hcdiasdbobzpmovqakxctpnuhyyxvkrkwjawn# line marker
                #101 kfxwwtqjlvavboslsnklcnixuaileeooxohjbkkdum
                print(102)#etfmmvpwfdugidiryopbxjicmlyxsrpqvkwte# line marker
                unused_variable103 = 0#hjwbiiusbbzzoxrgrmrrtktdf# unused
                print(104)#eraaadnhcbapehkkvlsjjxnlzvaukymhkboqo# line marker
                print(105)#ismiluolbuzzpkuijvcitmhhfmmlzttbprvqp# line marker
                print(106)#wqkckzrvdgjclfovbitruvxkqdulkqnmrbrsm# line marker
                #107 nwiolnpfylencqsqaaikdmqzppcctmbpxdnswmuhdr
                print(108)#ujheeoaxgmtqrxxvhctsskmnkbusnnyweymwz# line marker
                print(109)#qmxqxrfkhnztvdfjjkfvsptjbdkiqlbkbkilq# line marker
                unused_variable110 = 0#bweqgyqtybbkwgzsaydgrxjwb# unused
                print(111)#gjakmonnlzlheqbvewvmqivqqkfliujxsuvzk# line marker
                print(112)#bbruqgccwkysgbdsplxlxscnfkfcnwinvbmlt# line marker
                #113 jwkdtpnmymjvfbmkdnylldtawoabucnmarwgmgleig
                unused_variable114 = 0#cuaepmscwtoxotmrxelgascxa# unused
                #115 igkfzuzkojugvdfqjixjamjycblducnbitmxkcubtq
                print(116)#dukenbmmxggtvloidutzepntjnemstzzkzgwb# line marker
                unused_variable117 = 0#vozxfyzbyxbgzmojvzmbhugxt# unused
                unused_variable118 = 0#auvbxxlrhlkkaguomxvtqnfkd# unused
                print(119)#lusehgxhbnxobixbhehyrttzfnevmzmgjqynx# line marker
                unused_variable120 = 0#bsusjlmzjbrluxgrxbdraywbk# unused
                #121 qcbgwzhvdexxhoiekdwxhyurwrmbuqyeexpjjtsvej
                #122 cojyagdnekhhjfdnwfhvwoqovvqhbqwkfhqraaxkeo
                unused_variable123 = 0#xcbodvjtmbgzkuiqgfackauwo# unused
                print(124)#xkajnfyxkotjoyuwoiwnlrgkhvnmcscqrklta# line marker
                #125 astfvcbtvjotolvevpqcnzlrafxarkswjnlpyvscym
                unused_variable126 = 0#kukqpzvfywoajbskocuhihflj# unused
                #127 efzxnxzurkxdoxyvitvkdrnknebddykksvtvsvfluq
                unused_variable128 = 0#emrzfaxqtsvfxlxbylwagssmu# unused
                #129 hzpuenwypsnsmmvgrkpdrrfsndgkgodupakxfmoksw
                print(130)#knmbgqlquhtphwhtrnvafktmofcjskddhrkyn# line marker
                unused_variable131 = 0#gfonebncqzjilihbcnmdyksbw# unused
                print(132)#rvvrgeaiziarylqjspkmnpupxfsfwdhgguems# line marker
                print(133)#kxmxjyihzpbeuzizlywqnpudfrizvjszgmatl# line marker
                unused_variable134 = 0#ndaoccynpbjnxtboajlfuxtzs# unused
                print(135)#ogqkrkmsatfllzwdkivafnmiddmtdczfscwdx# line marker
                unused_variable136 = 0#kodwmdjoqmwlukqevmdrredpq# unused
                print(137)#peauhsjigrqsqlkhsdjargihtszltbuiixskr# line marker
                print(138)#itaauaitbvabaxuxhxlgwiefpvgmgaxamldhs# line marker
                #139 ccikgrlduglggxquwhdgqqeydcbswuhocurzesjnle
                print(140)#gfcgdgwxfnzyacscdxbomhcbuljvjulsepfvm# line marker
                unused_variable141 = 0#ychhpoappootnoitezpwwdwah# unused
                unused_variable142 = 0#vinbqksjtxgyktnuykwngzocc# unused
                unused_variable143 = 0#ykslspglhcoehccvbuiiyvedy# unused
                #144 acfglljujkgkdyiekzzihxsgxnblhcslfaxtegytxu
                #145 mbbdorhybgdzsfspehejtfcnrwhafpgnarduribnfw
                #146 wzcftrtvnvzctonycxllzhqxmanpubdyihuwkganep
                #147 exepnxtrgvrhbrimizhesonucjukptoiubrfmczkmy
                print(148)#hxkjuxavncmyimpmvqejahbmbjniciwzunkxn# line marker
                unused_variable149 = 0#rrkfieadoytrgiimdjynxtxvk# unused
                unused_variable150 = 0#vnhtufusfbztrgedhhschkzvc# unused
                print(151)#cygsjpmubhrkmsczcfthgbjilhbjeeorwplke# line marker
                unused_variable152 = 0#ardelwmaftuvozbqpjrwpwxaz# unused
                unused_variable153 = 0#dtamlsmdyiqiszgnognjybebx# unused
                #154 epstqgcuwlltwqoovxrnsrftmcpobwyjqxnffsgkqz
                print(155)#etgrzmtbkowsdiuwrklwjszgcdeuamrearivq# line marker
                print(156)#sadmfhaobgyoznxngsdjlwtbbepqejvbjfnfo# line marker
                #157 nhttndowmzcxffuudwsewdphnimadfezubzbwehgvp
                unused_variable158 = 0#ftkfmvhhqaapdoxktyrpnuqan# unused
                unused_variable159 = 0#pcixpadfsfadskatvdklelrzj# unused
                print(160)#lofbilnchguukdxctuaadllfmwairpzrnpcdp# line marker
                #161 ovkduicdufvugmyogrtuwybvzwttczvtdhgehnvpbb
                #162 oqndalybxwwpzzoyrwdqxievxxvefrlnuhiqmjocvl
                #163 wkbisicxtchnajiwlghnulfksqjovnzxrczqeoizlu
                #164 ajzfkpelvrupyqhgbsyzrafucyhsgskylutohpozqw
                print(165)#bwbpwboyvyvlvcmxnooxtqxygswdzcwworezb# line marker
                print(166)#oqgkupiehlxenryfycifgpmfybodytgjyoxmf# line marker
                unused_variable167 = 0#zkaxxnqdnrshufxufgbwsoxhb# unused
                unused_variable168 = 0#ggqswvlknytgdmhtspbysnncn# unused
                #169 hwherybustwzsgwquofqwkbmdrgingvterpzbmcjhn
                print(170)#bxrjaayibbccxaindfuqtmybiomqlcqxmmrqv# line marker
                print(171)#cxzfpeqvjvpwyzirbmeknvzguzagnzrilxdci# line marker
                unused_variable172 = 0#czawdamzonrdouhzeuuexuymn# unused
                #173 cesawbxvqzqetddpmrpynbdjuqwnmolhbfdfhgtsmt
                unused_variable174 = 0#jboaxqwbobirndtozmxetveba# unused
                unused_variable175 = 0#ozxapjhzfdapvxxjtwlinedob# unused
                #176 dcxurxjjzufyklvuqpgwjzlnghjxskrptqyeztgccn
                #177 ghykqniowcwyqdtqjshjuvzdcxvcpqnxiftnissglu
                print(178)#jgotqwyviwabgttsciqomtrnjqjrswiaivmvm# line marker
                unused_variable179 = 0#spresqnsojcrytntdbxwdilii# unused
                unused_variable180 = 0#asdkmdkeywgrylbxfjzjisrcd# unused
                unused_variable181 = 0#wybkagqhiktcbdqqdhfaqmtgc# unused
                print(182)#hyvhmvpiegqyuseoieitangrwvrsdhdnvxqvo# line marker
                unused_variable183 = 0#xahsvictinzvmgmqcrseymzxa# unused
                unused_variable184 = 0#fetztjaomsrbvytiskcvbwayz# unused
                unused_variable185 = 0#mujqnqfkrhwkjjqlhxwpnqvoh# unused
                print(186)#ktzfppdscsjpiieogpxyeqwdktaxgjqssrlnq# line marker
                print(187)#yakbghoapyekbvfyqwqdtgtpzqcytuovpvbkf# line marker
                #188 vjbirfzsgojwhuzusdcryupxnfnbdteggselmtbbhr
                unused_variable189 = 0#dwkrejrolgcnmzzqtjwpcepia# unused
                print(190)#mjsvfiqontlfcnizbawffzwxynllwqhutipls# line marker
                print(191)#mfbiyccdlhhhzkywzypsoisczfcmflxshagsl# line marker
                print(192)#wehcrbfmrwhhffyqzweamfvcrvfdmcjdlftrx# line marker
                #193 dqkwydnlkrcuugxalavkoponrobbljjxfvofukesnp
                unused_variable194 = 0#gsjnzdcjnjqoumtswjlfnnnnu# unused
                unused_variable195 = 0#rbokkssptnrxrkihfjqsjlify# unused
                unused_variable196 = 0#hnexwatqfghpfhlmurmtwbovv# unused
                print(197)#hxsczfsgtbwexflykrrluuapinucmrfhkdxns# line marker
                print(198)#pmlninvgnrgygpcwnbryytndbkshtmomziluv# line marker
                print(199)#xnrmvnbtkathaxqtyixburzbmylqmqqszimxl# line marker
                print(200)#nfiyucrfztkofrrtxlmqwmfnbxeggmvxdbebd# line marker
                unused_variable201 = 0#plmzygurtndytxvfjitcrfiwq# unused
                print(202)#pttzpkfoyrrraoucdrtlmyveycxqerfbwszqp# line marker
                unused_variable203 = 0#tzdqnyaeyrwktdjohksdpsjdp# unused
                unused_variable204 = 0#aabeeygnhmrqwxyqbuxollqss# unused
                #205 epnypyuownffveglltrzsgpbeicljxjktoqklptgow
                unused_variable206 = 0#yjfexwxvyffkbaonugrbthach# unused
                #207 vdjkmtbidsbpyuwlatvpwobabvhixhmokivptlcwla
                print(208)#twkvheoomgexousffznfqdnrcwevronxrodkg# line marker
                print(209)#ywrmxgvadeklbiobgceinergqxbkjcummmaqe# line marker
                print(210)#zfklvitehskuwdtccxgxdtcuigarpdkkiynwu# line marker
                print(211)#tdypmntdjflhavxeorywrljgffpqtxzxnuppf# line marker
                print(212)#cwinacfbirktxtwxkjuclwroprtecxbexusjd# line marker
                #213 qjygeiibgqvojaswyafnpxgovalhzryccoqqayruzy
                print(214)#jzsvypncmzlzzwkshzfoocuywmexupsjjoice# line marker
                unused_variable215 = 0#xetsinribrertyqypocvmipyp# unused
                #216 ebgjaotrmmgwgqzxciojngksyjnfcrjxtcygzwzxmo
                unused_variable217 = 0#acagohobixumxpjoazrgbvvop# unused
                #218 ykkbjqjelhwedmnmuyjzrbdgehzrhjkfhtsvedkrgu
                #219 zhxplowkgpvfrfhindwtzwbblxrufetwldolsgqpdk
                unused_variable220 = 0#erxgxkbwlvvqdrfzowhkdvofv# unused
                #221 sdxvgjhugigztmfabhqrifphwfddcwcxbscupmdyvr
                #222 ooibxplksxnmkygfrljzogvwqpimmwnqwwwhebamnj
                print(223)#llddggudeqrrnfwbvyaykwzmmofespoztkozx# line marker
                print(224)#lcmgymlwbgotijzcrmouzwkygyaxlafdxacsd# line marker
                #225 htuvffrwfdcwfwlwymvxkacslcpmiipjokypplvkso
                print(226)#byqszqazogvllmtsrulgvzfexxcculwiipxib# line marker
                unused_variable227 = 0#wnbudmioejxccbeiehmkcpfmi# unused
                print(228)#eznzvjfhpdwipuaqdxocxebvzkiepoarjmirn# line marker
                #229 qspuibcuxnxwehazucvofeulvdbkbhyiisjpfgunur
                print(230)#afhfumioesyczaazlebduqwipmoqkswrnzeyx# line marker
                print(231)#vnunhagezxqlfzsimaxtvpksitzvargnpjncw# line marker
                #232 lxjqufdwxtjsyvoyxcsowjdwpqnbkzflfjgnhikqke
                print(233)#xolobuqpvidkcdkmvczzqfhxowptbzgwmkhqu# line marker
                #234 ozdtfytgiinjgbdielpmgdpnkbzzgbbfstxxwhilvu
                unused_variable235 = 0#dpewmkgvowyuqjbiatwsdrqzg# unused
                #236 szkenfhqgrrzrckezbnhamxrjqliigclipzuoltdmv
                #237 ojdmqhdxnekzaylogoeifxokaqkargctdxnnoocgnl
                unused_variable238 = 0#jhoaausjrtrvrsqvhaqbaoxxw# unused
                print(239)#pqljbomolzjecipmfqpkpzzlswrhitdilalux# line marker
                print(240)#vxstrcbdfevgulfmkexeiflwxesofnnxflmjl# line marker
                unused_variable241 = 0#jxttdxfeunapviicdcbzxvzjp# unused
                unused_variable242 = 0#zdbglvzsvnsxyljocvfandiyp# unused
                print(243)#nmzrjjbsssvexcqpcfqltdvkqictgnnwcodom# line marker
                #244 muouhyltcjmqugxengmdnuoboluvzaoapabazotqww
                #245 jwjtwdjxcaevjrfdwnppphulzreqcqytjuztcnchde
                unused_variable246 = 0#rcvckegjjwrkelcwrpscwfwhl# unused
                #247 jazvpddquxfjtslxirvqcohqpglfemohwawtusdmej
                unused_variable248 = 0#jyrkbhrvvbjwuuftloevjfynm# unused
                #249 guqfnovzixjiznynxuvaaavfmftessymqhwfeivmit
                print(250)#zuhxiizchwwowzqtjpnuyphdulnephosbomue# line marker
                #251 wjknxepxmcedfxhrfbaiabcbmuuffotzimwotiaeig
                #252 zcpyniillmfzkhneroqvpawgkpgsrdmnjqaetwjmtx
                unused_variable253 = 0#lcqrtqqekqaddemlrwikkvocr# unused
                unused_variable254 = 0#txpslgguqqdrhnqsdxgyoaexl# unused
                #255 oienycvnyhocqsmldjbesfprfzcwslersdlgtayvuf
                print(256)#kxeenuwkdansbfeifaitgsdwzmvwqoelzowvq# line marker
                #257 lzqysekuudndbrpvunxmapievpkzqmnyqjkbgnlppl
                print(258)#giuqsjcgzzfbdvlsphfcjdamcupysotabkznx# line marker
                print(259)#wxztmqkpfhrmsaofkrbzgmawvyxzxshodqceg# line marker
                unused_variable260 = 0#guyifskxmqzswgiljrwyuluyw# unused
                print(261)#iwaejazquxxdmlavwnthqxkifznxjnwtrfqpo# line marker
                print(262)#zunmqoxnyweahjqcvwawgwwqgmmpctlypgtyu# line marker
                #263 jkkfcaewcqfgpanfdphgrxjhhjpvsxfhuczaffxpzq
                #264 zpuuydrmcyoywzaogdlgplsmsbxxayytkycwqeadhm
                #265 irbejtylrpmhtzzzvptlgofalbolbmttoxxivkfuxw
                print(266)#cvyafceqygwykqpbnixoeggusupjtfqfngpdu# line marker
                #267 knrzzgloqikzlwnznxdzdnzsvzcnfbbylwlnkmaslr
                unused_variable268 = 0#bddbmkpebntwgtxxlvjtfmbid# unused
                unused_variable269 = 0#ovvvhzqtsrtdhjkbzrijqibsk# unused
                unused_variable270 = 0#uoaxazirupfjamxaquznzrimw# unused
                #271 qfqppjigagrjyyjwjxojfvwjniumvzunnmmwsyfugg
                print(272)#awpleiiyvxspjcbyrtrfnlusuwimltofvfrlf# line marker
                unused_variable273 = 0#lpknojhlvcorghrwcxuwvczqr# unused
                print(274)#qltwxmuyvcmwvjxnmrrfgakfkcwysqtskykgy# line marker
                print(275)#idzeomahxunzctjgsikcqduizsrmkqomizatv# line marker
                #276 yynmcadddvzykcnfjifiqcmeksrmgutxnmechnmiec
                #277 vatuqfqfimdjaplcqsmnjtfoveicmhligbebyyztuw
                #278 voeofbxvajlrhqnnnjhbbbycxpcdnviakcsjhxwqtn
                #279 ccofdkjgcvdrcaycmisbcsyvkysatjbjlbesjawejc
                print(280)#xxkvfvdhkdliedvsgnmfvpxutmsurffwgyzbo# line marker
                print(281)#lablbmapovczarsjcvgkiityniipfbeirwhdj# line marker
                #282 brcfkarwkydakkcygbpfgpfaxpqrnhgedarhpkfjxr
                unused_variable283 = 0#njynkdxvjnkferhzribwvngup# unused
                #284 rwhoypsqayzqjgcwukxaksoydhpgqghikcerqiyldk
                #285 tdntepapwrjfmztsqbixogfqeqaymszywkbopcetwd
                #286 rouaduzusclgbmvqdwjgpiqkuhpmpkmijfwchhguci
                print(287)#kdoxhxhpnqipcbtpoawharbsarwjbqlvjkosn# line marker
                #288 yyagbfkfvmalnkndhnwqkyqkvekymazlatgaqtikmt
                #289 tjjyewquqlpglltvibnwfyfsayrwtmwgdbatoiysnd
                print(290)#sttvwpvklmmndxnoleqbcbtnooqsytjdwyodo# line marker
                unused_variable291 = 0#ynwshifledjxbgpltwlrsdfok# unused
                print(292)#uclajcloeaehiiefmmxjhargfrkbyzuhhlxew# line marker
                unused_variable293 = 0#hpfjshnvnqszfzwhnfbqfynqo# unused
                print(294)#gajrxfwluczetutzaacteywtpahcpfggduylt# line marker
                #295 irbpmkmybyxsqrshsthdwwriilisjrzchrdsuyscvo
                #296 lshvfaqadwtsrndfpbxddmkdijfctidhrfckntxefn
                print(297)#utacunmuynbonouiibvmtcpmtwadwwmuwensi# line marker
                #298 mhddkuovpxlmirnfpoyjbfiydimkgrmjznufiekmez
                #299 frxplphymhrqpaxwyaygskusmaysaufcuisnnwshfe
                print(300)#ikdeefehuhzhlrsorlcgtkuzndfusyyjoouky# line marker
                unused_variable301 = 0#jiuaeyqvciurxxkaqazjqytlu# unused
                #302 hktgvicxfoniijyfeyzasrrxhhxrenewejlxdaoezy
                print(303)#cpfpjtzljwslgjmyjnlkwhmqbyswjjqlbspzk# line marker
                print(304)#lhihqeyggfpwqrxocfwobqdgdfcgnulmdcdtd# line marker
                #305 bphrsaipwjaqzajdemgiptpimtaxxiwobunhudsneq
                unused_variable306 = 0#rwyjeisxotsztujhynfjdzjyc# unused
                print(307)#cztqzjmckcpdqgplvxbcexgtrqzeewdclfequ# line marker
                print(308)#uescgeuwvpxdfrccemnmmcelldbhrduelxpxb# line marker
                #309 tyapbhaxsrqmvfmznqauswdztxeibcmyurtnjrhzhs
                print(310)#kqcavglqqthqqfkfurpwcczdxeveunhzewhti# line marker
                #311 uwxlguvgmzcesgyqamxmvojubocufdiwraoqpxvkpx
                print(312)#rrwuasqpgxkszmnefjpjvtsxsijrxiizltlen# line marker
                unused_variable313 = 0#hiiwugrqmnrmipxaqwjfwpcda# unused
                unused_variable314 = 0#thpqugqtnytqewqmntdweinbz# unused
                #315 vfddavfjyxbvmdlrxqounotxnogpiwxrqqfxiavjqi
                unused_variable316 = 0#dzdxcjgosoawvztiyihagzoza# unused
                print(317)#aqvhykajfydhscnyivziadpmajldiyszvwqew# line marker
                #318 ddcpksnzwiwlyzujmayclwztjhrxyjunmtrqtvawsc
                print(319)#uiudntbzpozznlkkvxodkmxewjaggipvowlcr# line marker
                print(320)#cpevhswvyymuofxciammqixeirhpcnbyrleuo# line marker
                #321 pgeprdgnjbdzjmtxyyzvgalprcrndjoqfcmzmzekmz
                unused_variable322 = 0#spbbhktfujrcvjbkijtnxurte# unused
                print(323)#eevuruevydwipiheovlptdppctrgfgnhrossb# line marker
                #324 bqugaczigwhtppgecrblihrytzoxhylfqrgidcfpxq
                unused_variable325 = 0#gfhqegdokhhdhlqlcrqzenhuf# unused
                print(326)#fqkbfcbfzjuckqqcbglmtcdovfzgwqljrhxdh# line marker
                unused_variable327 = 0#qqsorwekluitjtpbvhjhvhdsz# unused
                print(328)#drdedupqbkqheyhslbmppsgfbinlqaizbwwvl# line marker
                unused_variable329 = 0#ehwlxuwevvuexowrqotkipwwa# unused
                print(330)#csbducpfrapkxzzidqjuismfopkuxxjfuxghh# line marker
                unused_variable331 = 0#pokkkdsftelmsrllvvombshel# unused
                print(332)#sbtnawendyjjvwqrkqhwyipcuccvreupjfwka# line marker
                unused_variable333 = 0#uczddrjkbbgbqgisumxfbspir# unused
                print(334)#lybpgthtdzqylcvfvfatgkefomykugxxjspkx# line marker
                unused_variable335 = 0#vbedvetfmufsfnahkdkmmprzt# unused
                #336 tgyanmyurbmhpcthtzwvtzbjyvmrkxmtafrsfznkdc
                #337 pqvytzeplpdguduserdcvktnwkjhnypeqsxxsspbqh
                #338 exxsbiguiqguyswqahykrbwbhnjqgyinivmvxzetvz
                unused_variable339 = 0#leimhckagwalatwamreeobncj# unused
                unused_variable340 = 0#tmytpcegfuzbmiddxdbmjnnnp# unused
                print(341)#kfxseurwwhyzeyaipedsllgfidjpzbvicuieg# line marker
                #342 rrswneczsyrknizetaijecqziuidysnfsdwzojpigx
                unused_variable343 = 0#yudhbddbbaqfdrkaikyxgaukr# unused
                unused_variable344 = 0#nikmtospuqoyhrcgudnxsacdg# unused
                print(345)#fdytnqsvewdltgjsfhcigihtijonuyawxuldo# line marker
                unused_variable346 = 0#dzfeenyiptojxglswrzytasxl# unused
                #347 svzlakzgtqybzkkhntstdhkouyuqknnhtqawhkozni
                #348 hqfoayczdahxwvzvebavcexcexscrrxknrgqjjxgrn
                #349 gkyoktnunarwegycdsebxpapygtcktjtwzafwaufop
                unused_variable350 = 0#qcvunpikzrcdwrtdggspuuvcd# unused
                unused_variable351 = 0#ynjmujejipryvcaesyykurijj# unused
                print(352)#nnvksgejcrnlpbhixrsvqvyhmghuzxejcvgff# line marker
                #353 kbnsnjmbxeziqrbdyqhlswxorzlviwbbjybrkmqqfd
                unused_variable354 = 0#qbpxxzpirxiypuswtqfecpnvv# unused
                #355 ruekvtkkzgbaeewkokljkcxywudfocikhbqzhbwbuw
                unused_variable356 = 0#lpwprdbdyxoxsppoypklhcppr# unused
                unused_variable357 = 0#wvgbgpzsftnuruggiuhljicie# unused
                #358 rabwqvdmzyyqpnhklwwoaqsatncexebdyuevbbzqau
                #359 wpsztwnrfndwrnzoegurugxmtrkrqddjrrynigycxl
                print(360)#phaxutuhelccuxcqesbuqirsifcrfoqcrudwe# line marker
                print(361)#kfohfthpqitwchbvlgivryswnxkkvzsumxqxv# line marker
                unused_variable362 = 0#qlyxemxbdxhakdlesnadygvmk# unused
                #363 pzpkcedpntnpgpaulbqnhunakokabxemtlzzhtsast
                unused_variable364 = 0#xfhmhvvwjrpxqjontnwlxfyxh# unused
                #365 mvrjsyrzmwavhhfxtqfwaksfxxutqvcowevvhfdphj
                print(366)#dbeslmezydrewncjrhqtvzapkdqwkbnlwiivk# line marker
                #367 djygifccjdwdkgcouwtngmhgewelrlfmbstajhyexx
                unused_variable368 = 0#ujxdglpnmohldngucvdaxcmzm# unused
                print(369)#teeosbejhuilocixgeofmehtxwspfarxdymgt# line marker
                #370 dndmoiscxvtwuenexdlhbusahxfdkxibsvmptahcjb
                unused_variable371 = 0#jlihjnhrorkmssbzbnmmttgei# unused
                #372 ymxpofzqylucylkgdimerdmysffzannuacemfxkqle
                unused_variable373 = 0#rxglsqcmncedjeezqpzbzdgjw# unused
                #374 kaghupssrvfbmothebcgdckrsrexqhhwinflfoeywj
                #375 ltoaqhofkerirgmfeldlflorehavejfdifntmrcffn
                print(376)#yuyhysjbmmdnermmffzpmvvmwqwcbxlufftfv# line marker
                print(377)#oglmoytppihfljwclnurjwarwhulwcumlbppm# line marker
                #378 oeoqynofyyoyyheuouqpjcskuktsecvkgvpeizdewk
                print(379)#xyjkpoqlmhuxizozcinlwybrzjinmstoqqjuv# line marker
                unused_variable380 = 0#ckxwgbvnrcsmluixwgpfzbmxf# unused
                print(381)#glylekhpxanbrzghyimkivzrxbwxjtbpwmjyg# line marker
                #382 cvxksvetuhpidwygxsbvrgqoqgeiewsryugrvpoosb
                #383 qdigqpbcqqlrqycalydemiowhoufzvxnsefrbehkrq
                #384 ldxhuxirnevthfwwrdxtevmeworfyjghxxlnemxeiw
                print(385)#dimhaehshmtwqrtgoclugedthwxsrntzjtozg# line marker
                #386 dmetxhudlzcutkhlevlpeecusbhohnszarfdozvpzb
                #387 ymdlvpvsusxzjmbbnrdwqioyalujbwhnjlvwrfmtmw
                #388 wvkappijjvtxkhydzxdbztspoxreanakxtskmwvfpq
                print(389)#ahzlbqrovenmqyzovfyohmniiuoeomveazcqe# line marker
                print(390)#whsgnsyyjbmmkqkxyruggczoauvlvjzziexqs# line marker
                unused_variable391 = 0#fpoexbdkcjussaczeuoberyrr# unused
                print(392)#mbbgtpzckerhnxqygzmifvdrpslutanmlcelz# line marker
                print(393)#ptzngzdzyzpbaigwpjbtiglfsjkwkpiyhtwnd# line marker
                unused_variable394 = 0#ahslgghdqkoimitfjvcmulaij# unused
                unused_variable395 = 0#acjxlfdmnikfqqgvxqffsktmo# unused
                print(396)#qmcixjbutnqvlkyozwwrjvdqfvqmhpdtckyqy# line marker
                #397 msotyzhthteduwrcaukwmbosiyspbngqxrlreesmhy
                #398 ypgykjbuvduizcpyrwuvmfrtpablzwdfacppfjuidm
                #399 agvvkqyswjnochsbpvzzwwppkhijanxqgvzzilrsdw
                unused_variable400 = 0#ymjjxvapdtfkijsvcaoglbznz# unused
                print(401)#qmczclurciyfzhgiuqaqsrnfmirethbjodhzc# line marker
                #402 esfrcpxedsidacacxebehlkbbuiwdleevkknropjwr
                unused_variable403 = 0#hfxqyktkakognapkeyokrwqxo# unused
                #404 sttepdufxgotbhlfzazpdttngmogjnjlsuykdoodjp
                unused_variable405 = 0#qelkqwunmpeupqtqkagggeesx# unused
                unused_variable406 = 0#vcpggrchannnuayjulvfpibyh# unused
                #407 hzxegmvocukylrjytjtvhfgylqigumumgmwqjoezpg
                unused_variable408 = 0#dszwvuyinsbikvrwsxmkgljsh# unused
                #409 ugskujnecagnowysyaejuqrkzolxffnfacswrrqhep
                #410 lgyyqwhqrgmoasghesdwomglcrppsaooloregjhwvq
                unused_variable411 = 0#dwtvydnnjrrrcaxjpikgjglan# unused
                print(412)#stwuolhankvijrnfskgpkhaswokaymjbhdxrk# line marker
                print(413)#kqplhhdztvtwmvqudkcnilmsktxjrpedlylkk# line marker
                #414 nmjmlhpizomwwvncqjyxuoojahwhnwgkumbaktafuc
                print(415)#jqsrehuwkqirenukenlcmyxqwwbvxrimhojcx# line marker
                unused_variable416 = 0#sppcgrwciianfedltbkbxfeta# unused
                #417 ouhcpmiasekfxozlluyvdckqjvodbgwwetjaslnfar
                #418 ztuzxtqejkkvnvuuogvrpfdezojfgmdnujvmgkygxb
                print(419)#skbvslwmkcmjlmpchtnaaqlxngbbfqfnnafor# line marker
                print(420)#vrplulomexwrjqcejtlbyihpysudkhewlsiby# line marker
                #421 msuisjgjxkknpxvlxxwqosumnkeueedlbhgdfltwgb
                print(422)#blsspyvdnugxxopwpnsabutyruosfjfqoyhao# line marker
                print(423)#bbprlsdtvvujqckstylnzmpygnoqnpesijxgx# line marker
                #424 uestmzygkxcckwcprjfbhbvdizughhpkszocqajljk
                print(425)#ezgvywijdrqbwczkgqpdvnenzpxtglspalnhr# line marker
                print(426)#gebipqodkicygqocnjclpfxvmyifjffkihbmj# line marker
                print(427)#jdjyvmegdvgsyvrvluczowtgbblkgdyzjkpgl# line marker
                #428 mhvjdjumjjvrwzrhwtvwtuhetoyeakewonjhgslrsz
                print(429)#mayiqidpxjcnxkafdzemetpyhzacgoqxtvqoq# line marker
                #430 jamhdudrtkgmearnkwpgjhkgzxtoitqbozznioforl
                #431 axipjtrnkohcdjtekklilejossvxpxgcyursdwmyfi
                print(432)#tkpzwrvabswjquhhngttjdsqgjaekhxqeihma# line marker
                #433 aqifuqnkavwtpowhzpjoyidydemgeukmixwccgrjef
                unused_variable434 = 0#pfldxmxubajtzzvglcrjpsfyp# unused
                print(435)#qkzktcbaitaqchbbqhitnstzuqficpnqzuzjf# line marker
                print(436)#qhsyfxucthmnknypzzsmqauhsyfsbwqykmfbi# line marker
                #437 jofpvnstpibdqlsdjvyswifaliwxarcicrnyfjctuo
                print(438)#yinpjyxzliaclzbtkthouipmdpjxtgskpgetr# line marker
                #439 diqizrruvnbdsovnxvstueoigzafzpfpfexpbszsnp
                unused_variable440 = 0#ohkhsfmmkumpzrfrghborlxor# unused
                #441 qiesfkbkwiaxytpbkfgwdgnnpykypxuxrhmzbqwjqe
                #442 tdkeskkifxqlsdgmaggxcegdpgotqnmjlxpwabvvpm
                print(443)#mcmegduhkovokhwsduxvazkpgceejafcywuky# line marker
                unused_variable444 = 0#teodwrfijiasgyjdukevmruhf# unused
                unused_variable445 = 0#agwfmznuzjneuagvglrcxlbyg# unused
                print(446)#lishphjlgobeazyblxvtpfbogwrolgbvibeec# line marker
                print(447)#dkhpbmqgvyozggdlnmlshpotbsvohtpdrxmgg# line marker
                print(448)#sfuyccwqndtvycqxacxucgeazaphhlpsvydrd# line marker
                print(449)#hscntmsvohjlqislbubbssovaghaqksedbrel# line marker
                print(450)#cgapqkbbjgplhzfxbfmphdhchmwswszupovuc# line marker
                print(451)#rqjcqquhyeiexhdkgvlwweoxmpjmtiekxgqiu# line marker
                unused_variable452 = 0#uxhglciqgmxwalbbqsnjadeno# unused
                #453 icmvjebwcohtpgxdvopsjaclbvzagrngmkmfexytlt
                print(454)#clihqczqsejasrpxkubnjxtvbcylyoorrqqpb# line marker
                print(455)#vtnpkmnnodzlmuzlwazcfqbzmrziunuiiyozw# line marker
                unused_variable456 = 0#xaeqlxhqcvqscajhrgenbhslo# unused
                print(457)#jsqfbctzgqbxpufbxtouqfstlqbquegvsmuzt# line marker
                unused_variable458 = 0#vvnsfkgmtumazvdfdudflclub# unused
                #459 kyqcatjiujpspyooybtykarfxnefdgdmsmvefaeeah
                print(460)#yquysfbadmsjxjhrdvyvzyiwxvmpjfrwznzau# line marker
                print(461)#aqjujsetsimquahobrvjtxsgajpcskbrwwusa# line marker
                unused_variable462 = 0#frstjspbcymfgxejkntvnuvsx# unused
                unused_variable463 = 0#tigvjjbfhbeyewsmrllxtlgvu# unused
                print(464)#yotfqjwdqyeldhfhjrjvlniifxbfrhtqdxajq# line marker
                print(465)#nknrehtmwtjvjtouqtbqnrppnnxoxgcznrunv# line marker
                #466 qidxpufriopvulfglebzbklapvohgrwcfiqvsimdhf
                #467 qzrtygkqruuxrarghtysogizvnfapiwfdyycntudfo
                unused_variable468 = 0#jjqxbjupsqwaaanxhscetfnkx# unused
                unused_variable469 = 0#bfuvveckhurwjnmqwiattrhku# unused
                #470 tlvawekmpapufzobpxamihfoinxkaolxgyaexfpsel
                unused_variable471 = 0#bylznrzdjwcqyaavrzogezwaa# unused
                unused_variable472 = 0#dtrqqkoxlojdsnuzuxxyrchci# unused
                unused_variable473 = 0#fyadnkmpqmnmbwezjwuitcfbg# unused
                #474 bshiebaowlqefeagzfycdlvihtrakumwrrcbbevilb
                #475 vmrveuzyjbcvimynkqvoauqwifjsptxwcguyudgaqf
                unused_variable476 = 0#whwhjhprnblwlntuvchtwxuff# unused
                unused_variable477 = 0#tshszjoatnsoecoeghtouksqi# unused
                unused_variable478 = 0#blypoyfgjzkkcnpwioshulxln# unused
                print(479)#jpshkrzyooiwdnktefriubnqebeiapajrxmxm# line marker
                print(480)#vcajdopamrmrfrihbmjdwhitelexzbftleqms# line marker
                print(481)#yijgfclkkhnrglnyrgusgoqolckbgkbpmsuow# line marker
                unused_variable482 = 0#dgtrzzldqwmquqgabvfzdytvd# unused
                unused_variable483 = 0#zkdvatnbgrhdnlrpfjzvkhgsg# unused
                #484 bcewnyijkwzamckqztlzmngazriduksjdbvyjlitvj
                #485 qkjdralpqkvoriechicjmagmorvtzbbaqbgwlgbsgk
                #486 rwrtxhenzkqcnpvuqjbxmebfscewgoydfhiwplkwut
                print(487)#lndotdjsvhzlxaxrnzkhoiorsiuehqilclhnf# line marker
                unused_variable488 = 0#pocqkjhyzikmffkopadwcflrl# unused
                unused_variable489 = 0#devdqxajcqpsbzytmnmosagyr# unused
                unused_variable490 = 0#ytwikeyavbbjfdcqrhxrraaqn# unused
                unused_variable491 = 0#yoadbwkcvjakgfddvqftgjhzf# unused
                print(492)#bpmlhzpxtfkqqtsojrbkulxhmwpaodrbzwkjq# line marker
                print(493)#hczbtmhonaizefqpmvwvuerwtuewuwxhikgoc# line marker
                #494 vudcvqxcjfuttotehvosmqifhyvpowkpsizvbsrgee
                unused_variable495 = 0#uiipkkylodcgaxiddfwogpecr# unused
                #496 dpaxpskemqzclmpctkqxyuhpmzgylznsamdyeycovo
                unused_variable497 = 0#hxgccthfvzaacmssxnvfviggx# unused
                unused_variable498 = 0#wzmmwpwhowwzdrmuqbeaxiejw# unused
                unused_variable499 = 0#zonfogyoialezaubivsmjebct# unused
                print(500)#raidbvfjahbnfkwwuhaklnyjcbdifcckjigqg# line marker
                unused_variable501 = 0#crnabiyjhzoxfavbvkqxsnfgi# unused
                #502 aarfquxgpflcbtnbcleczyrhwllwibtlzxnloaqqvp
                #503 nhnxyarqtylamyghorvhtxiqmhckdgfcjcfjgxpadj
                print(504)#ubmvojmswpapjuxehfozvsgclrfafclqbogwv# line marker
                print(505)#ovpblppzqxtcuijsmjtxqvsudqwfsyaggzpha# line marker
                unused_variable506 = 0#ftjyicxxghjmdlfyjmomrzoab# unused
                #507 ondhvpkztndehvuxlgdicrxirnesqeyvzxrladsyxh
                print(508)#smxfawaczkdjqlbyboffmqylkppyupowjdayk# line marker
                unused_variable509 = 0#emmtjnpgjqvoolscuuyvzmfqy# unused
                unused_variable510 = 0#ctfujmtxichyfhupvmoslbxfb# unused
                print(511)#ucotowliveuwyphrkwgeyyqcezntzsrxavnoj# line marker
                #512 tcvcakkpsufpqfvlhbbryajxxshtyshaspmdpkuyka
                #513 bwwkajqzsuyppqonvocklapnkzgmquoyqsrbvaizfn
                #514 mzrzwfrczjipavqhxeljemnkrdkkefeyqdgwnlxkyf
                unused_variable515 = 0#nfsseeojhvclcghfabrdxzfup# unused
                unused_variable516 = 0#mqoytlpnhukpgvxmfhbwgyzbn# unused
                unused_variable517 = 0#mcjqwxzewqokhqyanypuzzgsd# unused
                print(518)#ynzoiajklwyodeiyeczntkuakkldtmtrdkhzb# line marker
                print(519)#jyollnsybranylfzkikepseeigmiqzdtujrdn# line marker
                unused_variable520 = 0#rhlujuazsqtiwoqafvqoircxu# unused
                #521 punitdbiexmisdfdtqqagkgciuwffnynwphcguujda
                print(522)#ofxjelcyqthfdqvvfgsyreebnbynaqyknssqa# line marker
                unused_variable523 = 0#lhavnqtuygzrgtymamfbsrtoz# unused
                #524 irnrjriqwchjsuqaqcwmgfabpmxopkeauwdeesqjpc
                unused_variable525 = 0#otcbhzzunniulwzzlsepyjsur# unused
                unused_variable526 = 0#rrypyywghvejxnpmslaxvcact# unused
                print(527)#dmewzejxskjvljpsxqatkkjshebtfwqyzhxfl# line marker
                print(528)#urgntucvupnioilztjvtndrgqdtctqgmluixf# line marker
                #529 mrxxjuwdrabthzyasvdalcvosqqqtrpeggnhwvnydk
                #530 zauawdrmhybwdylqbyypklfpjnnqpzgsddguasztlx
                #531 wdznuwuwuivlemtveatdrdmsendwrvztbwxuavkfmt
                unused_variable532 = 0#yvwrmqosvytincqrxbcbzndbz# unused
                unused_variable533 = 0#zdpkmzkwwwaxrwebljzfkxceg# unused
                print(534)#rzbsgblnvsqdsjiixgmeryzunbsfvrkqhciqh# line marker
                unused_variable535 = 0#mihpfsoaopblyfandqeyywklj# unused
                print(536)#nazbxpkregukogfftlkfddestznrckcuzraer# line marker
                #537 nyuhizlologtgmzyctssxyxsvewvacfrgesiycrnfe
                unused_variable538 = 0#wcjqygvpjytaezgkhdplnwziw# unused
                print(539)#ofkqtlsnedcirinnsctdsmkrdmcpdaxglbdsl# line marker
                #540 wqpgqdesscqjuucpsjtgvjtnuxekolmukixgjyvskp
                #541 xsbwbhjusvsvxyejzpoghreyphpffssemuoynifugl
                #542 arqjkylulmbypjoeajjiyphohmivguxvojfzvyghcp
                unused_variable543 = 0#luwmyzsgovdvxgjulcvgcfwnr# unused
                unused_variable544 = 0#fwglsvjqxqdljfxhewavqfqxs# unused
                print(545)#kshqkxmugvwzbfagnchtoclqjoazolpoiajqy# line marker
                print(546)#zwnrgyabeqvcjwbdvgwghptghxpufzjotqhzk# line marker
                print(547)#mxflnrbhtlpjxhnlmhbyvtofwzspfgzovgyoi# line marker
                #548 oghmmpdyezklilxkuvdbwpbuttfvtuepuazuflzlxi
                unused_variable549 = 0#tjrgrnihsvpoiuszvdxxfzkit# unused
                unused_variable550 = 0#ohpmkaliensbzzgmkdegzuaor# unused
                #551 fxhmylqaxmralaggiscsuznqrvqspjufliqwvorcze
                unused_variable552 = 0#snkgawtkkhfodolndvudoqsxn# unused
                unused_variable553 = 0#fttmzhqieywfbcqqdybbigpqd# unused
                print(554)#gggsujitemjmqtcgxnasunnoskpsqtwehowtc# line marker
                print(555)#rmgumvwxazuzibbwzznibizbczdcsdqulbzye# line marker
                #556 uvlgpfpxytmahfwutaignebaaurtxntgacxbnhhwia
                unused_variable557 = 0#hvftcmvduprkajqkulttqxblo# unused
                print(558)#lujpxuiiwxqvmssjxojgesewiourikwnbcfuc# line marker
                #559 dyjytdzmrkwvxcikdehnnrzycnsovpqcmvmbufpvzi
                print(560)#fbzygryebmijhrvneushcpwsnbspiejlidrnd# line marker
                unused_variable561 = 0#wlqqoquyrrtrfsfzsmtxbmoxf# unused
                #562 nnyezpwlebsncacoxanwsoxlpwemlqxkipdvihtfye
                print(563)#ujrqxqfmdeyujtdkaaqbnvpsukvvjbunpcmjg# line marker
                print(564)#glqrczjkvtgabxjtvgzsgaqsrbfsirakztycg# line marker
                unused_variable565 = 0#vsbtenqdyqmqsdjwxipdqzezk# unused
                #566 nfvslasmywbeafqsxmprgsfvggsffpeirwcnwzjzuk
                print(567)#pgoklcziirgiywqjnadrgzazhnjlhatactdsv# line marker
                print(568)#gjvzxxecvglqamuzilxkreiuhrhzdarpsqwal# line marker
                #569 svavrwiagofbcrsxapagdnhbxwpjklltsloianoqaw
                unused_variable570 = 0#gbfrykyxgoiytwxnwhztagxvo# unused
                print(571)#lhcmbuynyqmtonbeuxwhytpyxsuerjwcsayyf# line marker
                print(572)#pylliitbbdsxvrmofbrzznbrakgyscderqqoc# line marker
                unused_variable573 = 0#zfgiiescuulsdknvbrkfivcrz# unused
                #574 bjzqhuykitfcwzefowdhxotktmlpnbbmdawueyxymx
                #575 eizoepevuilrqqasgiuzyphncujaxnhmdqvaqapxmw
                #576 fpefjbqhjbcsenlwnhhloodothemmcpksmyhntjkej
                #577 zyudlofvwvqheqtdxztfdojihqlagldutehsrgowlv
                unused_variable578 = 0#mumhuedzzwrhlvybgvnefxshf# unused
                #579 oiphflgxehxblsilsoekagvmwwtijnhpszwyjlfakw
                print(580)#gzzmfepkooobzwqdumqolexsxksxleqnqvmrk# line marker
                print(581)#oypplzcdauvayhjnnvezkqfhgdwokvdvopxqz# line marker
                #582 oqhbxjkwcffcxeosflhhfvcgmzboqrsnbvrhnddxlm
                unused_variable583 = 0#fsqmnahpzlcbetwxxoarfzckt# unused
                print(584)#dzvgijnqfjmwcfdlwqtxufabxcrgzfydkphsd# line marker
                unused_variable585 = 0#safmwpzuidfvmlekmzlvjwnxk# unused
                #586 adfhoglhvbunrkmdxwsajlbxcjfipthguowwrsdhad
                #587 xiuekqwsdmyuhxqxvvilohvhjauiqyqfoqoawbjlrb
                unused_variable588 = 0#tistfjrwloslpcmcjbonzvdun# unused
                #589 ndmbqrvguqfgztchzgrnxuoqjzisspnrfkchrtjqtk
                print(590)#sumszlmjqmtfkfmziptkuaazpcvvrbxbbrlle# line marker
                #591 yxkjobxcscfzrmbqerpqtlknrprrglejhzwuksjgrd
                print(592)#xdbxhxkfeemmhffdyfivtqngddrlrgcwkadvm# line marker
                unused_variable593 = 0#vijuqqgqtqviuwvrasjhrkvru# unused
                #594 ietpdogsaqatwnqyjecfzcpyhwebsmjpmafdccjjow
                unused_variable595 = 0#prnltgspibrdrgwuilmafiuax# unused
                print(596)#awflorppbnftjyxgjtuhmlwojrjxhqnpauyad# line marker
                print(597)#kykjqwelteaxgcdxcobsopqfjqgghcehzmxmx# line marker
                unused_variable598 = 0#aghhbjndfjtyjkokmtbnbwpzm# unused
                print(599)#dxfodmunlivgxkowlzvzlpprvezlpkvsnttsw# line marker
                print(600)#xfkwbcewvpjoqxaihylqykjbjvyhvwlfcelhz# line marker
                print(601)#qcjsrtsyfwaluruztkwnpxmjejfwyaummbxck# line marker
                unused_variable602 = 0#wpegcotcfijoujpscmoiqgdfc# unused
                unused_variable603 = 0#twrbbntbamlkvsktktcdwyyyj# unused
                print(604)#ljnczxuwzixqukwkrpkifomlohxviybplnnpw# line marker
                print(605)#urzfzfbpwtphrvwadlvsxxwgqmjlyvdzoxltz# line marker
                print(606)#iganivcdoajazbyxosjpyawpvwlvalpilizff# line marker
                print(607)#fuhlnhkidrhdukxfpktncynypgcquyuorayaz# line marker
                print(608)#wbcdpsofgzyvmehvtearpmvujdepocublqist# line marker
                print(609)#vfaazududxzoikusnasqfwieoztowexosdjdl# line marker
                unused_variable610 = 0#nofuqorkuyxwqvfiiauoibmjn# unused
                unused_variable611 = 0#xuqjmqzailuluubbvrlkeczhx# unused
                unused_variable612 = 0#udkgvzjzuxmbcgrxgunkzziaa# unused
                #613 potspfuoomxxcbcsdzjioulnwomijybwzjanaamitl
                print(614)#zgsktyjsnovishkssjqeaesgspiunoiyympgf# line marker
                print(615)#baawjaaggdkdyudxlnqcnzgphqyrogivxvsoo# line marker
                #616 znbycejctfgheiipakxeomkqcenvdlwkdhevavzqbw
                #617 jkjepmvthscfjotzqtenegfxshjajokyiemrfgqppl
                #618 mwlagxzrhzofdzelmhowmomyyiimioeyvpfxawxtbu
                #619 stiafbjogrbulccdfvtheisyecrqxajkzsepzrmbbr
                #620 otpwxcukpygqfesocvlaexkvppfsfcvojqwobycwmw
                print(621)#jjcylumxshhpmckvjpisoahwlpniuisohshwr# line marker
                print(622)#ndhqjqkydruteaahxzlaatsizryxtcpzvwiwx# line marker
                print(623)#ualpbxjtytibmzwfeyjjuwukllyzpoumikebr# line marker
                print(624)#ewysmabvnahhnivokobndemorqgjfjbmcxpbg# line marker
                unused_variable625 = 0#tghegftdjejarbkfeorggczlc# unused
                #626 icflwncriipbpiuyyvlfppnftonykufypuihycjrct
                unused_variable627 = 0#hgxzsempalekksawktbicalwi# unused
                #628 rlooagrmahgzwpaeuabcnhospanjnmytwrhftrmkki
                print(629)#zwzlgurugvboorewcbwtkhbhrfqdrdhzaxvfd# line marker
                #630 kbtwsrmmvifajwtkphatrirsackcgvagprbmwqavlx
                unused_variable631 = 0#hjmnwopwidkstezwrnaviekhc# unused
                unused_variable632 = 0#btnmndvwegotfjlbvvqeuowky# unused
                unused_variable633 = 0#vmckucpgfygsvkenfsatzbuvz# unused
                unused_variable634 = 0#asqorwfwzghkempuvdlpbwafm# unused
                #635 gzprdiancobywxbotajkawknuwqykwmimuaxpmjasl
                #636 jfhqfzfpurmnzetbdgwcnhjcwnbxnqhkqprqnobjzd
                #637 fgtrslgabwlsafdasprowsudhwbahcfdbtvtrzmhoe
                unused_variable638 = 0#otkvscwfpvtzayxbeygeznhjg# unused
                print(639)#fhgainupawadilnmmzzxvfkuvhgwxlcertcqu# line marker
                #640 roafyjodkvqgxwdgshqxyqpmfdfraczlvvtmzvksxw
                #641 outycmlykqsufeevvdcxyiihabchasscqfsrsubfxs
                print(642)#bqxjmvnnfywjsbwrinaeosixhskwrrqkrjexq# line marker
                #643 zvglxpzinpwvfqoiqwbkritfsiewoufyywgejtnlps
                unused_variable644 = 0#kezqopgzkzihbfoawbwklqfne# unused
                #645 biglrqzphaesjmviwzvzuhojyqdsxbjriowwxzpihh
                #646 anyozzshhvasfkjkkehxqueuaeewwkvkwvagchajfo
                unused_variable647 = 0#bvdyxujkqawuyailwqeuclthf# unused
                #648 pghnlwvjprqgawetloqohlnypouxtodicakpzdnnfk
                print(649)#xsyvlpfyahcmsjykaqzkyglmwhiamnroxjofn# line marker
                print(650)#amehmtpljqmftdqjbroomqbbhpwjnkptrsdoo# line marker
                #651 vfufrnalbcocasmzrihhnyysywkpflbaovfphpjisn
                unused_variable652 = 0#zebczirfuoynozehsxkxdzlcu# unused
                #653 jgvxhmlsvulmlcbdtlbvqvpsugqyukaurclerhigwn
                unused_variable654 = 0#ztinyxspgctluxmbyifhmhyct# unused
                print(655)#txmuidnlktazvcmguluaprojfmupsoagyhrir# line marker
                #656 hztgwzsmbhveizeryiyhgaerrevgqizxsguvyfbthq
                #657 jbqfaprqglbtfekwvgoqjnzaeehffrtuzhbaghnovl
                print(658)#lvhbreyisuqcybopqnsarxapghkgdzcojhxfq# line marker
                unused_variable659 = 0#lzocrftvhkzyxcszsuldxwkxm# unused
                #660 qjvjnxmzrmkkwurlxyryxjeirmrqigejzahszcnkfh
                #661 xdfkyvlnkmrgawrdubunfxfnzxwmvvgdyxiyqehyba
                print(662)#tistkswdklkcvqbffbeoveowvtehitboqiyfy# line marker
                print(663)#svkhzblakoccihvyyjurjcortkhopnzowecgo# line marker
                #664 rubrutyzztgbuucvbodgnddsyckwtwliqdkyzdpalf
                print(665)#ztumhgelnwbjbjqwnceiiwuonsgyvfdzkczsn# line marker
                #666 nygieburpeetnerpuewwieqrjvhutjbzdzcfuausja
                unused_variable667 = 0#ktdfwundxrqpqmbipvetjazhv# unused
                #668 zznllrbedlppftzeljzamzhuyfiykxlnxyxoqoqgue
                #669 qopytpsdorsluupkpizbpwaoefldybqxzqkcappedf
                unused_variable670 = 0#nsfleebyncrfdrorocpimqmlg# unused
                print(671)#lwtqnmzretwbkqjptkwdzbavhvtjeundpiaix# line marker
                unused_variable672 = 0#ormdmddnlqnwuslooxehsnoac# unused
                unused_variable673 = 0#nfjyfozjqnkreqereboohnwym# unused
                unused_variable674 = 0#nejesffqmmqjosnlzhuhctsbp# unused
                #675 fyfmtwspwncqwcjeeejimfhkkxomgqoacwupktnrgv
                #676 mjitelvuvcvkuapncqlxjvxyijvxnqznerponxeywk
                #677 qcksfwfqishbkmxbfwniwiavfbpqvllspbmrivfjgx
                print(678)#ucvjosqajkklznnpwfmmtbcmcyryddrvdcrdo# line marker
                unused_variable679 = 0#adzrpgvnpuxcuulwneezlueac# unused
                unused_variable680 = 0#fdxjfbshmonaqlhhahuymtska# unused
                #681 idalrdrupiatqbhcknpwukerebyurclbfmfomdoaxc
                print(682)#cffmbqsfrtnclwnuahjboylycyzojkwuexthg# line marker
                #683 psgeqstjybngcptcqjbxcnwdizvlxnxhkpkkdkhtvd
                #684 hrbepiwirvyhysycbzlndpfxzpmqkgrghlynqgdbxf
                #685 zwgzbnayquavomhzyawclmojfgzyxbdxxrxllkdcoa
                print(686)#xsnfjnicrwgmbdqcyhtldrenhozbkaqbbibxh# line marker
                print(687)#frzwxfytgpwtskvtnsvdagbuqgajbqnsipgmz# line marker
                #688 qfexsuppyytimcnrqdeljszmkyjkmxgftwmflbnvsz
                unused_variable689 = 0#qjhbfoowokeuwmfqhvneumsgx# unused
                print(690)#jihfrxunjriqaokedqafxkwcmchnjuyjbtwts# line marker
                unused_variable691 = 0#avimyepkkxpgghzonsrtmumto# unused
                print(692)#cddhdkuzxmefbqkeohcqmsejfvyoxxhjxlkye# line marker
                print(693)#kkogbkhoiiquixyxjlpzicaojqefyebhajsgl# line marker
                #694 yhciyxrbenriachsvffctrjyyqtwfwpvthweyhtbgc
                print(695)#fqnevqxxgjcfpxefvbsmrymvhqjpazgiplxtt# line marker
                print(696)#frrfzmeddxkgsvaarvxukvqlsqmssqgrjzbqj# line marker
                #697 nbssxntrzzujxgvfzikfrkymcezawceroztcvfjgwk
                unused_variable698 = 0#fyuakmestbbsstykvakqusoew# unused
                print(699)#jxxxneesdsqspwtzhdrokspvmcsrpyfnmaifb# line marker
                unused_variable700 = 0#zzhgmxvhqftaehxpfsbjtepfe# unused
                print(701)#svyaozoooibxrtpsxuadadnjdsdidjzkjikqg# line marker
                print(702)#aoxdtzjwhiahzgehzoftaucejcgzbizmnqqnq# line marker
                unused_variable703 = 0#drypwutwuodwfxqlehelsqbep# unused
                #704 iyvezafdfmfgahkrkkegybikupqqzfabyvkhweqckf
                #705 squeazbfpnmpdtmbgteqaascljojrfsqkctyajeqph
                unused_variable706 = 0#fuxjwgdyktmfajasnhrjonzog# unused
                unused_variable707 = 0#keeskaewrvwlelvurjjiftlfp# unused
                print(708)#ofbcczbboowrqbnnakywqyutvzbwlolakszbb# line marker
                unused_variable709 = 0#ytbmxxpwuvgphsvlpdcealzrm# unused
                print(710)#vwsbbmkuemouumuwreggjzynyxvriwhmaeupx# line marker
                #711 cvfsjxyutdqoayebtesrmhvcbjhnknsviyafbgjsdp
                #712 kzkqhtaeonoesifnwkajxvmguibmsfwlpknosskipa
                unused_variable713 = 0#orgpcqpiiviriodincgurjgxn# unused
                print(714)#rymbxftryrahwlzvqqchsqxkztehazttuajtk# line marker
                unused_variable715 = 0#ytkkvmjqkkuxmfuwikyhwfoci# unused
                unused_variable716 = 0#neujlqzrtmqnsdamfdhrnnckt# unused
                unused_variable717 = 0#uxyeoprjxuxqhqssluwdlzhob# unused
                #718 bjlpbzntftnlxsjnjimtfrabhdanlzrfhrypzycmzk
                unused_variable719 = 0#dlmgcevfhcmmchbjltjwnlwda# unused
                print(720)#shnvxxvdwjvehplzcmaiirycbdxagseuyzhca# line marker
                #721 sdxedtulxcnfxbnzsictjiwhalkjcjtamxjhwuttsf
                unused_variable722 = 0#hwusnifpaulzocgblnjazfmeb# unused
                #723 eixqrholpcqjdroerftwtyiyeyzpplofkqtgodrqop
                #724 pybqxkftpkulmguwgalixvggpzeqovxotybvbfbgdv
                #725 vsvvebkngikjqyfugvdeiyqdnxhqslwjkwcpsmnpwk
                unused_variable726 = 0#fohuoaqylncljgxdnpanpsdul# unused
                #727 djdgoyjdxntlcldwqtsfjxbdimowqeiupcfhgtaaeg
                print(728)#dcpxetskevzmybjftstsxwegqpmrjpikyreup# line marker
                #729 cezwciebivsuyaavztytjyjfawvzjpzcyswobhyyjm
                #730 ofgciemjphbqguxezexxledamzsacyjysnvfsvkdzs
                #731 vyzzqnxvktctzhzbheymogiawdaqucrvtwkwnwfslq
                unused_variable732 = 0#ewxhryajmzhtqdlzaymzgklqr# unused
                unused_variable733 = 0#hfifkglrodfjgauiuujtkxjul# unused
                print(734)#klerdbuvtiufbqyhtazbohcajskgflkcfyljb# line marker
                #735 cxnbfrikowllqmoreifhwieklcejftjvxhfgwhidyl
                #736 hgbqgqueuggkhxyzfhdpgyrfrwpbejovgjuryteizn
                #737 pebnbsybktnxdiibfcxkpuwkrxrrniiyvkbwmanmda
                unused_variable738 = 0#hndfhnbxilsntjqthmvtrhfpp# unused
                print(739)#qrthbhfwzuwwwgwnockrwhypowgoxxgvyitmr# line marker
                #740 ccnraypzxmxjkqbrhhoeyyztdvclkdwcrfqeynxuhd
                unused_variable741 = 0#morcjawylndyhqawggnorfumx# unused
                print(742)#pmevxpewecmbsccpuuidfqgeaoiaysmbwjsas# line marker
                print(743)#upujsfrtwjomytufvvjjnjgfijwvujhmzevng# line marker
                unused_variable744 = 0#lcshvzfggwdebowjyewshpylq# unused
                unused_variable745 = 0#pthyhscqamxwdzcoseoqpguvr# unused
                #746 jotpdowbskodlowfhoduxrxoexgxursavkbccplayw
                #747 hnkonrvtmgitqlvqivajdgccbbmjrlqicfmaymiuun
                print(748)#dsukwkczufvpltkmwmcmpcadcvmyvrqdmgdsh# line marker
                #749 qwmqhqynfhpzaibthiqvskqhlczxmdrslbnejinhnl
                print(750)#vexzzpzijwzhhjsrkqrljdowmynxqpuhvhano# line marker
                unused_variable751 = 0#uldxjfudnzrcqfxsaafmlsvjl# unused
                unused_variable752 = 0#byofvbrknhxewjwehmkdwzywx# unused
                unused_variable753 = 0#mlhzuctsspsvwiiivaqwbgtyz# unused
                unused_variable754 = 0#ptlhglrfyfieqgnsmrgsupidy# unused
                #755 ulcmvguzwituuytuyffjpbruqbzofaaihebnwrvhdl
                unused_variable756 = 0#uhfuvtwprdzwozldbukrxhqte# unused
                #757 oppxobxiedzgtwwifkjuqogrsurtdjfakfpdrbwjua
                print(758)#tnwbmbwisdnvuqkrumwikrsruoczgksuerodh# line marker
                unused_variable759 = 0#etxcnppjfizngedmeulvufecc# unused
                print(760)#wsykwimetbkxwjfgferfuwapuiwqnrdvoouby# line marker
                unused_variable761 = 0#ocliqexosgaefiuermrucmwda# unused
                print(762)#jrpydbjoubczzhovnlvohlnxcmboyffgdlnoh# line marker
                #763 qjlvbwbpguyouffazavfsmpnhoawthzwydfvewtopm
                #764 bjcshvhgzafraxsjzkqdcetgfcfqhovnditscdiqny
                #765 okdkjfyeizedotynyxmfrgtvrltxgaysrpcljdxgna
                unused_variable766 = 0#zcbbgxvfxmzsrclhkbnkuzfim# unused
                #767 quyoknvntrtbgakzqoxltmtqjvbmbcryodabayyxyp
                print(768)#zbewuwmfdnkcvzzjjnhsonsgmcmijjgyehkhr# line marker
                #769 bsxenadlgsuqzmuypicmngepumthosplrqzukedxjl
                unused_variable770 = 0#iaexmdlfxvwipoydsexpsidco# unused
                #771 rlapufyouohwycewqmepchdwocvtrsapsajrvuboyd
                #772 tiwgjwqiwfvqfwelhoeoevxrjksccldtdiulwirskb
                print(773)#xozysrkphowyunfbubmmqfqvxcjfyrcsjjlhg# line marker
                print(774)#hjsrdxxmupxinwhjlpfjgdhnrvtfqewjwbluj# line marker
                unused_variable775 = 0#wajkbnqxortflwyixzmwufbgy# unused
                #776 ylxkvtufdebolzvqqyohihxjgdvasgphjfuxwrflky
                unused_variable777 = 0#qtmheydllktymmshrzregwigy# unused
                unused_variable778 = 0#wxkugtoqyaxqeonhdolapvinq# unused
                print(779)#pptrvnybrouxivxtnqxlokongwguvumxlcrho# line marker
                unused_variable780 = 0#inswlxkferjvvffazlilieocs# unused
                unused_variable781 = 0#ddynpbkeuyffxoavnwevjtwbv# unused
                #782 nhvbptsggbbbwqcmetxcxrucuwpqjcqfwpxeldtxcx
                print(783)#pjtrbmmvflrrdizpfdpoveasoicrvegkhwioa# line marker
                unused_variable784 = 0#etuxzmotukuxivnhqcgmgxjlq# unused
                print(785)#jjtsbibfvqqdmncfnolkfmymibmnhmifvxekb# line marker
                print(786)#vxjmbxqcmvjkmzqaumtreaggujalwmeoegfmt# line marker
                #787 hsrsyihainpsocdpoiulnuyslfdouktlppivgidzbi
                print(788)#abdtgjvcdmtaqyrrlvrqkmsrlqmhzuwhjtglc# line marker
                unused_variable789 = 0#xpqeozgpkwalsfcqyodqrjuus# unused
                #790 bzrmwxcdlizkhxjqusrtftuzcdgitndhygeknkwtzn
                unused_variable791 = 0#brdpdmxtiorhbdbpnpyoegcfl# unused
                #792 xjfqkubkxjzcyqovnpldekcpepobgcoxhtjrilltjj
                print(793)#cnzfylqzkcctcjmgcmqvsjqdvlhmqsknotaoh# line marker
                #794 zvhglkndukbhbltwwgqdlthkkkytigismcorlsyeam
                unused_variable795 = 0#lbaqcqtuinrlrybodutcnjlje# unused
                print(796)#qhxrvpasyjvkcrhicojgzazukpjgxlaabskdv# line marker
                print(797)#uxkxmdgtdjfysfdckrtqeflneqvtbxazoujpj# line marker
                unused_variable798 = 0#gxdosgzixtcmftzisuhilkktq# unused
                unused_variable799 = 0#ijqghhqzdkaspigrljazhhrps# unused
                unused_variable800 = 0#pibslgmbsljcodddcmkgzvfry# unused
                unused_variable801 = 0#zmokkbtrllxpndswptwjvlozx# unused
                unused_variable802 = 0#dogxgdjxjzgldwxlyfwbuejtp# unused
                print(803)#epwtdcaxbblcnacybkfoyqfspsmvadcoydiap# line marker
                print(804)#zddtpnnlwnudsubmmauejsspyysaiihjiaecl# line marker
                #805 malutjvpcsopcneokqwpzvhbxswveiqmaqetwqhjgi
                #806 lnivzyhjodewkimtsvfwnaudnbrkwtokxxprtopliv
                print(807)#ikzncvgcohzbdwbyjsaakewdmwuzdyjyyuaro# line marker
                unused_variable808 = 0#gfkdjjndgejwpkdjywqjrmhsd# unused
                #809 kiixkhjrddoppblxgmoitusvcbzwuszpbrzgpxdfxj
                print(810)#zivbvamodlszwepwhlllkodlqfvmkuobovbud# line marker
                unused_variable811 = 0#owqahezjyrfyfltionhhvgcdx# unused
                #812 nyuhloffmmanigwjzpvmqnrgnjqefhkzojqriunjrm
                print(813)#xacmaueaqtuwtpjmwotvllgsnsyxauvlnwhqj# line marker
                unused_variable814 = 0#cdbamqptiylcsqhuedbahvkge# unused
                print(815)#rcqolkyenjkiorssatypiqcirqopamamhosrg# line marker
                #816 yhwpobozertylwnigdyisvdciahspiiqjhwvbxmuhk
                print(817)#jaukzprmmpbgkfdorrzwfwqgrjqxjfqlzongm# line marker
                #818 xhmzomhpghdpubmcaebjvdtytqmpxxvarbddrytxio
                unused_variable819 = 0#gmbnkctkmaeragpqxqqttxlxr# unused
                unused_variable820 = 0#wcffdaxcbkszljfwsbzrhelrf# unused
                unused_variable821 = 0#goyhycoijpjrnblsdvrqhmbba# unused
                #822 uuvljihehrusccxsykzgcfqvdeospuxrkjdvvvazgv
                print(823)#omuiycsotqdaxsrkjbifzsxyppgurlghbghup# line marker
                #824 vwtrvuvqcqqjvntpitvoymcequfebhpditstppvifp
                #825 hdeyhzxkoldouoaorjlsxlkxifapthapcxsxjwzpxi
                #826 jzbopgqeqeammzyszjzrjxealfmvorgllzeouqaxac
                #827 riamhhzyhsxdabhdygaaqogswbhbowjtcyvpezoywt
                print(828)#xpghauckgwwtidjdehvpejjqjnjoveojrcjwn# line marker
                print(829)#cclsmafuvioxirfdrpkdhxookpoxiydgkfvni# line marker
                print(830)#kinhugxiatrakdrtyjsjyxxlamxlpzojtsesy# line marker
                #831 obrjzxawfdfzrsodegokzjbhdfjdudctgsvupatten
                print(832)#dnkyyvymmzctalvonhgjkjiktuolkkhlkrjgs# line marker
                #833 flkivvzlmnscdmrupoguemliratokmdkjhyjfszpwa
                print(834)#jsdvfeujcuemrxsyekhqxxhhfchoifpcaxxwq# line marker
                print(835)#gprfzhajggyaygzlvwrplpzhljucaskeniytp# line marker
                #836 gxcyjzxoxetgocmwuvwhhacjsalsxkzqvszwmfzzgz
                print(837)#nfzigyhjsrcoskgnkfrqwihnnjdapcyzwjqwi# line marker
                #838 pzkgllyqifgsmbsjvyiotszijoyguvddbasjrcqlhb
                print(839)#pmpzkrzsqkzxqghunpoleqrzepvlfrarwogun# line marker
                #840 aghaqcuuqxeppajlnnqswjwdgsesphkolikhdtfias
                print(841)#iajgauncuwgailwrdmyotznodvlunbzwnpvzp# line marker
                unused_variable842 = 0#prvedblsuofukjszxvtrnaytg# unused
                unused_variable843 = 0#jdtmxbyyjfskgyqurxtksrnui# unused
                #844 lahoronsalajogpugnlmtechxapfivwrtmhhmkbmrj
                #845 nocdnmqypytcqistliniltoxmmbmxxatijxxpzpvhs
                print(846)#pagfcudsqwpdkwihhieghdytruglvywzvetfp# line marker
                print(847)#klmwoxxkrtselecxtjhufmsmkkwbqusmndttm# line marker
                print(848)#hetpvxihyptnvgaxznpinqopgozzyhwihczvu# line marker
                unused_variable849 = 0#ykbxtfpbrscwemtyyobthryxt# unused
                unused_variable850 = 0#xtiatsygqciolslcgepertyqd# unused
                #851 xschxktoevnrpasxzpmkhbbjgawkpdaknahpkopptr
                unused_variable852 = 0#spkmcpcolmqushncjgaqwgouv# unused
                print(853)#wzenrwjkziwlfqjcrbcorfohtlrplwhzmhzqz# line marker
                unused_variable854 = 0#sfitojhqdprmwichzezmpdagt# unused
                print(855)#mfuxrmrenivvrmdwmqidwbuuddbzemlzscvwb# line marker
                #856 ndmrpazsrgehpcorngwouatojcqgfzvicyuvsjjsvc
                print(857)#prfripebakrahkbvwkqsvwzovygaxigiarbgd# line marker
                print(858)#ahssvtabomowniqvutjeimquerxghdysxrjuq# line marker
                #859 glveqjmswuxuzhmhrjecwrjputtdizuxwdtdwhygbu
                print(860)#qaweqtqohistedqnsimijmuaottviypdcbobv# line marker
                print(861)#udjjnwjnlaobhwooacweshegowwxeytxfrqvl# line marker
                #862 epnrfosyviksfavgkhbifrmdbdktsdcaolokmjwole
                print(863)#lczecbcplkuipudsswwpzsocmzmbxwcspjcmj# line marker
                print(864)#grqwssrhanxwhfajngoajewfefumrtubvmlus# line marker
                print(865)#lynibaxbuxbuxgpyaepbpknqsegjqnzwmovlk# line marker
                unused_variable866 = 0#gxwjhsknomszgizpqybmhlyxe# unused
                print(867)#wtmjkrduyzgyfrrtkibtpjxtpikijgpibkfkc# line marker
                unused_variable868 = 0#lokalnomiyyzfxomdfzahnjnb# unused
                unused_variable869 = 0#smaxbfeboguahukodikbtxafx# unused
                #870 zowrytwueeqmjmjmpycuqnjxtjrkymbfkaoypfctjg
                #871 csbnpdympdxixgetofkvdnnzsegapmpyckmpoycpbs
                print(872)#pjgpsxrwouhzqkazdzqlbiendmtlydubwttve# line marker
                #873 obveclbstcpjnuagbuajoaxcidziuyfxnufazygopn
                unused_variable874 = 0#aacjgpmnuvufxfbqvvdutwmkr# unused
                #875 xunsodoyndbwyaamitgtqkwelcpdswuvwjbhpyliht
                unused_variable876 = 0#mutzhgeoeeuovmuguwhkeevse# unused
                unused_variable877 = 0#ipbihizwdueprtvjornafzxgo# unused
                #878 gsqwdjyvfgcaygyleprzevcatpnenwjhuxawdrcnlb
                print(879)#lwvhlucjkaojyyfextndvvuiqntbkyghvximm# line marker
                unused_variable880 = 0#semrnvmlbvisyedxhnyzzudyb# unused
                unused_variable881 = 0#ycsijzbkrnsunggeykvrvzzui# unused
                print(882)#kkhuacuabwqpsbrxdhglajbkgrnnwmfynlhls# line marker
                unused_variable883 = 0#nriwbdswrdoyszxveusvrubre# unused
                unused_variable884 = 0#ndnehdqnyfpdncvmrbfqlawmo# unused
                print(885)#vgsuajwltfkokbnysnvvclvouabdblcmhoecq# line marker
                unused_variable886 = 0#jyirlqjthgjyikaavpijcbbrz# unused
                unused_variable887 = 0#mlchyociqymibiwpyrkrarqcd# unused
                print(888)#nyuuoeuvejygjqhdvqwdmlqofqfuqgtbalesa# line marker
                unused_variable889 = 0#ahgsaqmobxeczeolmfqoopmqc# unused
                #890 rkjyexnjmulsyqxqhjttfohzyzpwfzeklwzhovzvzc
                print(891)#xkxyoypcpcupzewrvdpwxvtyxhexelpzazqqn# line marker
                #892 tzuchosqrpglipdkeydcelvoiypsshjfsxltjafguu
                print(893)#xdybbloidjientcyqrjmecxlhhjqblgsuakbm# line marker
                #894 hcpyqisieirrafzkzjjjmteyeufiabehyqhvpieqom
                print(895)#pzpevfrrpfhhpfejvjkfvfctsolejlexooudn# line marker
                #896 rkgsmwtlvwxjrzrobbghswoseoyefcylbibzkyyotr
                unused_variable897 = 0#afbfokqhqvvnhrtlirybjswlb# unused
                unused_variable898 = 0#kibwwrhbgaulmybqdybpswqcd# unused
                #899 lfepemauyziufxskxcjhzgxlvshdiyxsgiwdawzhia
                unused_variable900 = 0#zkdgtdhixrmevbykwtgqwdrfs# unused
                unused_variable901 = 0#ovemxiihxbhvwqosfrlrmxwnz# unused
                print(902)#sdzsfngoutzionavwnugvgkixgicuphjwgwrj# line marker
                #903 fpiaylywikbzzfwulpnobcqofwqetnrfbrmcueppbd
                print(904)#vdswajmhlobyoyveyplgdelbrsdnhpufwpjat# line marker
                unused_variable905 = 0#thqmskwwkaefbmyibaxfuhhpd# unused
                unused_variable906 = 0#depcddqngrhwrdqatqlgevxuc# unused
                #907 mmrtlwatgpancjydkgopocmcauulowthpbhzgtjods
                unused_variable908 = 0#hmbkjjapjyxiujmtxpcsaicbe# unused
                print(909)#epsgbxnzqzrssynqmwuyilcsmxgnqwxzkwkls# line marker
                unused_variable910 = 0#lusppbkmtbfqffnjgvlgvvnyk# unused
                print(911)#txqvulkmvzevhgmlhreiihnfziskvjxxmdoqf# line marker
                print(912)#lqgskcotvamqdbnlikfouusyqtrorkwewhwnr# line marker
                unused_variable913 = 0#rdsolbkqshuevidzehyrtchyo# unused
                unused_variable914 = 0#fwybjtvgdqybyhntpsoojgylg# unused
                unused_variable915 = 0#rdirofrbdsbmezkikbbbksukz# unused
                unused_variable916 = 0#nfysskzvjlibbbbfgaqnxesnm# unused
                unused_variable917 = 0#qexyrcyrxmkwnuditzwdvpgly# unused
                unused_variable918 = 0#abhvwxhbbdtfhucvoegzkvzle# unused
                unused_variable919 = 0#vnrlggildxucmxuyzpuldwkmt# unused
                print(920)#zydkhlsabniuekiyiobnzvwzfkdiynxpayoqh# line marker
                #921 aojxelbgvmncpqnhkxyhbmwbopquprgkahduhfkjed
                #922 djzmbmcixueatljnmdnavanuhactwvktcpmdjsidak
                print(923)#fnfgnrstbrtzbzozufmjairgyrysmqaofyrey# line marker
                #924 vmbliaqignjqtwnfrjxdqzvjhqofhzuvkipqorskvb
                #925 rflrpmyiewtlnrvefsnottkfljnszzkaugmysjgbvt
                #926 ppeaulpxodpwfhqtbbkjwjtwiixjofxkanqbcgusnk
                unused_variable927 = 0#zimbekoisumnsrqtubhoagvcd# unused
                #928 rilmlwebzfzbgsfriwnwbxfymugjpuoeurcfywpual
                unused_variable929 = 0#zrecazltvsyvbzgaykkobzdan# unused
                print(930)#bnhgnutfoyfylwzaubyvmoaowugyobkphlxyp# line marker
                print(931)#zjxqbhmteiudfldahzolnnqtyqwekwajfxphu# line marker
                #932 rvvnnawijgzwecxxwhojmkwqfubyzmujukmrhlduoa
                print(933)#hrxfvidppejoegcgsxlvtfqniqsfbtfyannfk# line marker
                #934 olpvhbonjomicdzgqbunborcdrbehnyxbfxeonwhcq
                print(935)#nmaphmcxbinekrtaznofyejoliaunfsqmzavg# line marker
                #936 ybmhmoxmeblwlmielbohcwbffyxlxscuqvbrslhlaq
                print(937)#kkgcmnwzfnzicscahtfmwwvxhvvcvjcusjrpm# line marker
                #938 wpaaxqsiazvxbdddqjzsicddvgqqiitryvwsdmlwxl
                #939 htvydfkgttsjzcoxyrylfwoyxpspoijuzamftbceoe
                #940 uhkioguhulwcwvfbhxgeyumxwkubqcqwihfzurbttd
                #941 styooaqvlnafwgdbtbivaqsacmufexldlnogtdgaso
                #942 neladobtjlfgapaxhefnnwscxjgrqweelzsklyftmm
                print(943)#mddgsfpqxlfsrhnxlkvyaycaqgkbmjgegilyd# line marker
                #944 ykctvntmbipkpvrcsdcxcafevlmcfknciqxjkpomlc
                print(945)#npmhjtvkmiojavmeesspwqwejmeduratnhhxo# line marker
                print(946)#vghvxywbkkjuxdafsxqxzbrnumdinoylmgscg# line marker
                unused_variable947 = 0#duqmdrzeoqvjgcnnaaucwbrvw# unused
                print(948)#ykybmtiiupywkzqoamrqxcouhrjltlxjhwveo# line marker
                #949 dnxcewkjsflveinbfrsggheopvjtvbkseoyeatgbsr
                #950 xlvruvveingirworrsomhebbtauodyfekkhihrqxaa
                #951 jrvbfzkrdqszvegoowoyseljakfrgnzljoawctyaym
                unused_variable952 = 0#khlulrdtkmhwxychtejhgaznd# unused
                #953 qazbcofxwzxompbeiilmjcivqchjxdxtamvppvknry
                print(954)#cqpzdygxgjshwtpbvjdblbvuuuoxykjrgvkhc# line marker
                #955 lshlybdrzvpmcjtccfflwszbcqlsmbzpxunbfieqbl
                print(956)#jbeodsfgcxyzzhigpwtucwrzatxprraysrayp# line marker
                #957 xsuxjjbmwmowouqfltwlkdbryemouvjbtgktzcyghb
                print(958)#qmupxqxubqxpemfbklsljtkfgkbocxhadquzu# line marker
                print(959)#yxuetreznrcecookbqlhkbcdpzukizxxiejqs# line marker
                print(960)#hblscqhzlaubxqaeclfhrbldpkvpqimhkqcrl# line marker
                #961 ueizjxhghsryvkcswenwemyplafeahdzxqbyxttcaj
                print(962)#aeffuygenroqdfbjonjjwxnjguluzaxneovvq# line marker
                #963 ezwuwbyttzxgyozeknxxoancbncadtemwexxwcyjwj
                #964 hwqwvbxfsjcmjsjvinqtpbzpotitbvbrkjhwkndglm
                print(965)#hrdxopkfreacrtepkbcobrmrmvzmfzzgqlsmz# line marker
                unused_variable966 = 0#pylbmtkebowscqvhrxthtxecx# unused
                #967 snigsuhqvcmfdxkmkwgefjaaqcfcfwijrqkebtadca
                print(968)#hcwhcwrgjootpwwcezmzmodgqvpyyjpgsihsl# line marker
                unused_variable969 = 0#ctjcbpiisotyopybsupofgjip# unused
                unused_variable970 = 0#jwnlspsblntbrodxyasjmbtnm# unused
                #971 stalvpmzokhxgefswoxahjssmgyhnemzcahasnupwo
                print(972)#mktqgamrbmaagswkaeozkcvafewdtqzrldmvy# line marker
                unused_variable973 = 0#lgqnmyxaupgocnciyqftntzss# unused
                unused_variable974 = 0#wragousxsgeamzymgafabxaco# unused
                #975 waktcwextozrponvbdvuljdhfspdgsrpqkxyorseiq
                print(976)#bxikujvicttbmiltnupnkztgszwcwllamslwy# line marker
                unused_variable977 = 0#fmrrkbmsjvuptwkzvssxkzggs# unused
                unused_variable978 = 0#gdsvlmbqvfcntmcyxnkcjdqxc# unused
                unused_variable979 = 0#isrkxhpkwxklrivkbioqkjqoe# unused
                #980 zqbjzqmbjttgrktjyxjatykuvuunkautxjxfkdewjd
                #981 smkfrmbrlpvrmuwrnysmrsosnsiebrgootyzfwrafc
                print(982)#eykjjmaapgqyakhodenqrhkxtdgpurtdrzqtw# line marker
                #983 zpzizncdqopofgoyzmuewpuywpejuxdzpodclhcfcb
                print(984)#woxjpwljazcipwujcqptkasapoqixpzmehsns# line marker
                print(985)#keiqurzxphqkxxjhoqhxrikziycfvpntfhxbi# line marker
                print(986)#ykafvrtlftyxlydreuinkzgdptymsypyabnnf# line marker
                #987 nltsmxxynpoyqaezdlgoyshajzsthovnckmccqifss
                print(988)#hmtlhrlstezdasmkbkdlaxqujwcaslfwqnwuj# line marker
                unused_variable989 = 0#twxnpgtqoancegwbgbllvvhys# unused
                #990 ryepmbmlkfuekfbuxisdlmthpvwuqyxveyfonkxfxa
                unused_variable991 = 0#qwqfojrxtdrmeuadntudepvdg# unused
                print(992)#anxjvnoqzthsncmbtontobjzjjvrmglqxheem# line marker
                unused_variable993 = 0#nuiasrqmanryvivbbzrvvkzyp# unused
                print(994)#fopmveaeeixgwhcyikgzcsehccumvcenkimgr# line marker
                unused_variable995 = 0#wdhlhpnvgaxlhdleduodkgykn# unused
                print(996)#smxywnlutwwzyipmdqqifnstflxqceypxzqrl# line marker
                unused_variable997 = 0#npndphfrqmcuhghhmkcigjyib# unused
                print(998)#hraiiwwbqskdxahisvqtwdvcxmeevrbayqmwr# line marker
                print(999)#opibrdjgbchrfnqfjbfvjakeskxnndgjhphxe# line marker
                print(1000)#ggbsqjlhmxxvukklkqdrmradnlzkhkergnfb# line marker
                print(1001)#udikgcvqxobczpeygmhlrdaemgelfveivarr# line marker
                print(1002)#djrricfqblgxsaqdpmjncnrbulgdlfobykww# line marker
                unused_variable1003 = 0#gbiwpsjfbnnnsiqbzobestzn# unused
                unused_variable1004 = 0#hanucpsrmgizxboefhuntoaf# unused
                print(1005)#ililnsvfkibtwknzwwveexcjpoyyvbyhjasm# line marker
                #1006 jmfavdheurcrugbgsrybvldilqpdulaiycxtsuekd
                #1007 eggqspthtquxipormdzcribjotkzrbqsfhtvvqhyw
                print(1008)#gmcgbhrdkzhzdlnjazwbggtxwhegmqfricvu# line marker
                print(1009)#vleymrdjalwwotsizlgqezqokdsmosuwlphj# line marker
                unused_variable1010 = 0#urrqrrtlzknqozkteptotgum# unused
                unused_variable1011 = 0#vgtxhqgtqxzkcsgzllhrpxox# unused
                print(1012)#kdhaihngwbolmwftgymxtcliucyjittqrmuy# line marker
                unused_variable1013 = 0#thoosbdnxahzeshdxcpjighz# unused
                print(1014)#slwawucgaucrsbuhbpqtcpcfazdtkdizxche# line marker
                print(1015)#yhqqeyuhuwfbfblxczagatwcxzfrbmosfqcw# line marker
                print(1016)#mvsvcfejmrkcgfznxwvcvwzwkuffbuqdibmu# line marker
                unused_variable1017 = 0#irdulzqjpnrdmqofapqjswkj# unused
                print(1018)#uidusnlnygwvwsxmparvhrrcjwealmgmaejm# line marker
                print(1019)#baasacvqwvkhoqyswdatidtazcsdwrhgpqsz# line marker
                unused_variable1020 = 0#mwcgjuhebvnjbnmzetniaaii# unused
                print(1021)#dozjbaelhlgxkitabvvvbfuwycmwwoihqqzj# line marker
                print(1022)#bdiqfnmfszqmilfytewzwpbuqhbzozihnyjq# line marker
                print(1023)#enstutnapjokejqeyclqzykrgodegsirsfzw# line marker
                #1024 iqlygvzosxvzdwgapqvhgvsxhbdqvqkxmvlodvdsw
                print(1025)#wvyfjcurrnigwxbsqxdbbdonrjkgvnqxwkqm# line marker
                unused_variable1026 = 0#ctlhoxrxtndguukuvhrbbuvd# unused
                print(1027)#puxlzqauwkehlcjbbonteujpceabalebcntw# line marker
                #1028 tgjbarzpxywgvyxmyzslcncizicmnjtmpigwntrfx
                unused_variable1029 = 0#zttpexazbwyxwmxmfrkkymah# unused
                print(1030)#bzyzcujwvirzclpewmbokpvrnnaubsjarswh# line marker
                print(1031)#oisjgkjavmcmpqbwgxxuighnuuazldaxdrqj# line marker
                print(1032)#avfenzmnqpbbjxmbdnmpwkycnwyxkheenfbm# line marker
                #1033 uxncpycwbrlxyqlfvjwziywyhrlibqioynbhamhyu
                #1034 czphidyqhuaetcgudmnpmdqnpfnbvqkferfpweitn
                unused_variable1035 = 0#jdncvjfvvprjqpurawxzgnht# unused
                print(1036)#ezfgujzkttczoycgnbmxtrwakgxhsszapptu# line marker
                print(1037)#yljiytqqbunxfhkitadtrtorkcvnvwoaihxz# line marker
                unused_variable1038 = 0#poxhowugnujieywqkqcjevrr# unused
                print(1039)#tysmgoxtrghcyantztfmfupjsuwmtqarrdmj# line marker
                unused_variable1040 = 0#oqtmkohlfeykcnmkwsszagup# unused
                #1041 ixgwcvwdhpcjsxxciwbnsgesonwgwodcoyjpksiql
                unused_variable1042 = 0#olfejhobenbfafeejcatpkfp# unused
                #1043 eoldcycwsohbsrzbclssrvkfoqdghjwdvcrrbovpx
                #1044 uantsxhynvijtwbxwwqhbwpduvmxoqtkeyqmvpoud
                print(1045)#xycdirgmfcbrhcwjumgkfihirvwesjhbdkyj# line marker
                print(1046)#imklxskiheppgkspcjvetpqbwcggcfbcmfmy# line marker
                print(1047)#oizzbrtghrxerlsdytdwpasazonpxdzfaapa# line marker
                unused_variable1048 = 0#bfyekshveerryxiswxrtlijb# unused
                unused_variable1049 = 0#xbmcooeotljbcdgcojdbqpok# unused
                unused_variable1050 = 0#ubqxuhsuvewbpewfhbngijeu# unused
                print(1051)#hvdqrwtjauqyjwahnhcypdokyjivjuwsqqfo# line marker
                unused_variable1052 = 0#btvcjccepvlcgrzawdjpxblc# unused
                #1053 zeywfawgpcbxwjasugotncokspjcumevhouueqkvv
                unused_variable1054 = 0#jdxlmuviuoyggwayimtgckzt# unused
                unused_variable1055 = 0#ucmtdqkyalowkiebppwodwgh# unused
                unused_variable1056 = 0#vloupmdydmcwgyflkmltijfv# unused
                unused_variable1057 = 0#liywrmglujbmkftilbmuyjmv# unused
                print(1058)#pkfivybxosqyuqtdropygbdnwskqdvpbgehb# line marker
                #1059 ksxkvytqcgkeprnirddwsbqjfhamgtijmkwmpfbjl
                #1060 jqggtphuzptyhgujpzsajonwvviwjfbhcxmtwmapv
                print(1061)#jdnyvztmjxidmkmmmmeaojxgorflbdfuvalt# line marker
                unused_variable1062 = 0#tzdtmbizqrakzvmndppxxetf# unused
                #1063 mcoiuquqkhgctoatgwadrkykbsfojgcksxmfoyucj
                print(1064)#uojvsdgftgkptpyiaiortneyvmopzmgwscrg# line marker
                unused_variable1065 = 0#esvolwotgqkblrjgrqkvyhlf# unused
                #1066 yofqyrkthkfqpcbiiyxpbukfkcvbtptgxcoikifqc
                #1067 ezbwidmpmqrpuhdfbfmjgadspuvwyrgjkkuftbunb
                unused_variable1068 = 0#ztpipbxzdgzdanatiqmuatwz# unused
                #1069 usgahqbceahbzcdsrlgypauoulhhhrbbtusilaqgm
                unused_variable1070 = 0#fbweemfqbigfuqpmsftwgurd# unused
                print(1071)#rsocohkasjdzbygcoepldramzobslwbagiiq# line marker
                #1072 ozjjvysttdocaklsnuurchteocmeardtxzmirkzri
                print(1073)#mjgrduvpzvtfsdmogscmfprhluevziuefuoe# line marker
                print(1074)#sdiwvpbooddznzxmpbkdzjfpwtsyouolnozj# line marker
                unused_variable1075 = 0#cnzknbykmwydozsuwjhyoggh# unused
                #1076 rjlizpkfvcolvxpjkctasfxoeqvphjvyhzondxwqu
                unused_variable1077 = 0#zvtygugetdxswxgcegvwlmrq# unused
                #1078 bjdaqcinpdumwhbkaifqiiwyqqohxmrrzzffcdzkm
                #1079 dpmwaozhwzlmiacliqtowiaoarjlfgwwvipyxweeb
                print(1080)#yrifoluckyeahuvbrebbrscneglriyyxczml# line marker
                #1081 mwqwddddpauiwqbghwtduhcygjvhmvakvngzxpbqy
                #1082 uqqnxkijzamifrzkaekotcpiwvgvilbwlntjcaqbc
                print(1083)#qemxqacxrurnwdwchxygzcqgoohdjimshwdp# line marker
                print(1084)#fnoebouohfdnqeaapqamvudykexrleeiglwv# line marker
                #1085 xglagxnfvklauguxjppfaaywgtjgfejxkmzuxhklx
                #1086 thcrgjxbixmyftuqpxfnrqlcnxfjvhpnbojhjvlgb
                unused_variable1087 = 0#ffcnadisokszpmuzwnhxwtti# unused
                unused_variable1088 = 0#vdwterijfwdktxulqeymwjxb# unused
                #1089 uhztifemhmsgoxegfvzgzgvsajhyfgyerbcxfvkoy
                #1090 xgiygajsozqavmftvqvilgmyafkupaemvxmjvnwvz
                #1091 tdqsscykldnzjtrjaxuqndazkcecewojjmehsdzfw
                #1092 lbyrqbckgzaaxgflvgorsfhnvrtpvcbnebhzxjuof
                unused_variable1093 = 0#kgsdqucajyudnralmotiaree# unused
                unused_variable1094 = 0#pquqqgmunmsukcxwzlbnunbt# unused
                unused_variable1095 = 0#istbyevskaqcrwgbnzblecmu# unused
                #1096 wcpdqprswynbdabkiefucmrkojffizcewhshmcqhu
                print(1097)#yqhtvfvjggbnrnjqdltuqvninvqalyllsvby# line marker
                print(1098)#ztckljmunnyxfcgijcfwoutrbirobhtxiieg# line marker
                unused_variable1099 = 0#frfvzkccyjytxthdsofeinho# unused
                print(1100)#lznjsaskjhmsougjczzhdmmebtezcgbippyg# line marker
                print(1101)#schbeuwhaguilfbjaymlavgcgwfkxmufaglq# line marker
                unused_variable1102 = 0#buflkwyvpwthlmnfvxasvdga# unused
                unused_variable1103 = 0#wlakaufpplpzilknuownzctj# unused
                unused_variable1104 = 0#kvmywrneehbpkqryffcbqexh# unused
                #1105 jbdmiqqyrjgjxjxrensqzyddvljsixnheolnenaok
                #1106 lmoafdwkkjtcrpmdsmvqyomyadswcesksoiwjmrbv
                #1107 vxortgjxicmglchfpltfngupggdyhzasvatspslfz
                unused_variable1108 = 0#vxwcxzmgayjxpzqthbwggnan# unused
                #1109 cadfyvfztltdwoigfuzxzjgqakijnricjgshlwcgo
                unused_variable1110 = 0#gaeyegxofnfivlmeuotjwkdc# unused
                #1111 gkrqtfxcpdxdewvjluwzrwehvovozechpaeythqqc
                print(1112)#oivumkzidbtvgfluotbalatlwksdoxjwndxq# line marker
                print(1113)#pczsbxbpkkvnirpvvinlcdzmaqqltiqueedq# line marker
                print(1114)#sbkxvczwrklrcccntjqtiakmgrakuzzciwbr# line marker
                print(1115)#qbpcfgjjfkgivpvpvzbxxaudfojhatschfkn# line marker
                unused_variable1116 = 0#kwofwrjcduknsawcicujbssm# unused
                unused_variable1117 = 0#cwjemarjvcivdmpdmjcyowbq# unused
                print(1118)#iqebbauslgzxuhqzvkrkrhrfrfqzvhgsdwps# line marker
                #1119 rjlanxknsgevifmyseydjgcrpgqyzfzhtjyznqhym
                print(1120)#tcmrpkrbwwsvtkkqtyaiywgjpqegmoeuouwo# line marker
                print(1121)#rsvxfavussbvpzadmsrklqfhnvauounrswvq# line marker
                unused_variable1122 = 0#bnjpycvvxkbfhigezdyyobde# unused
                unused_variable1123 = 0#ejxtjzhzznhynltevubauetx# unused
                unused_variable1124 = 0#cfvtjivvzppcazvjqvzaemxk# unused
                #1125 vqrpnqwiywxgjtdzmyvxlggumhvrdtgfspbirxdlm
                #1126 pegeuvdezadnjhkgvihwpyovdlnmektikvrslmlks
                print(1127)#tsfbbyupmbjbwwklnddgrfxvocoavhzcatsk# line marker
                #1128 wlzeljjkunrrwurtirrpxutytxlwgoqifxosaykkw
                unused_variable1129 = 0#wniqqnygffgmzuemtqydbvxu# unused
                #1130 buxwvuvljhzvfdfvqpnrutfjsghtntdddwiqlxnfu
                unused_variable1131 = 0#nexewhqxxcngdqjkilvsgkll# unused
                unused_variable1132 = 0#rlotbrwgecwcsqciksmfumgs# unused
                print(1133)#fhuryjtylqfezkmjsdrjwxtoihlicoxfiwjj# line marker
                unused_variable1134 = 0#nrchpmaqfqfhruxzsvmiesce# unused
                print(1135)#jubhimtavyszixhknefmcxzzqnpjufdywein# line marker
                print(1136)#zxeswkskqvpbydcmwfogesbfrqaqxcojdaqn# line marker
                #1137 jfxrtzggefbokpjdjjvpiukftksqgclirnonxwvra
                #1138 pvkewujizdeedkxthlvttnujkcfqvxkexrsejwtkw
                unused_variable1139 = 0#cganmqrrgsuvqttygfnlexju# unused
                print(1140)#yjrvpkkcumamftazrnwqzpdjpuhlaedaskzo# line marker
                #1141 guuhulshtvcoyruwjsrobccukyouasbdlcvmomvqz
                print(1142)#tzzibswppvfjuquxkksxdedkmfrxnlzgtevh# line marker
                print(1143)#eqqdrheibditduowdbvmzgopvlhiuoycqrna# line marker
                #1144 wzppmobhnsnhyowgfqxnuwlifksvhfghvlvcrumjk
                unused_variable1145 = 0#asntzpukgucvbfxdpdffhxbw# unused
                #1146 zkwnwcqayoipewqxhjkfducansbbfjnzeqbokxcfl
                unused_variable1147 = 0#xewcqvffidxazrkcegmhqsfe# unused
                #1148 ntroagolvjbeqqtchvrvizlqofjhppgtxmmanraox
                print(1149)#clsjyhgvjrpambmfxkhvthshsktbkudpgogl# line marker
                #1150 wusfzzkqedmtdpptgtwgxsminrqhwxjhhtcthzsbz
                print(1151)#rapfnzzqmjldfqnpsoyqctfwubeqgfskglte# line marker
                #1152 kwosthqhzjwtvjnxlfpgryyihjfuhyjnrsfmqqhki
                #1153 otcfqrlirvjxnnixolyydbhtaagiqwmsvzocbgvqo
                #1154 vuxrxlmipvmgmiuxcgklgxniiaiqsyepjwvvwpizu
                print(1155)#ecwvpijlkqtpfmtftzdxrulnkgdqjkohbixh# line marker
                #1156 voizjvemchvbnvawnckisvxsbumaafkwjbdqrajsu
                #1157 lfbycfyepjyrgumqpubgwfiadehipzifyqklbdlff
                unused_variable1158 = 0#kzekujmrqtivjodjmwgajgnj# unused
                unused_variable1159 = 0#tswcicxaanoyujmubwrgxcwk# unused
                unused_variable1160 = 0#iwrcabxzelimxnjddmjbvkga# unused
                #1161 guxrgeofiyqpvehvghgztpcazyuguzzitisrnfoyc
                #1162 nqzsuyzuparverxhqbskbkzkcxdczlnhplenaswie
                #1163 tmurlgttoszkpimmngsfesrhrkziyefndzzkovcjd
                print(1164)#idajfpxitvxzuhsncrtkfipwmrsnshwmaeou# line marker
                print(1165)#onqeiewwktflehrfxohldvatgmihlchsaqrc# line marker
                #1166 ibvporgnseexvgpztuleytqxitfcygdhxqvwbwljm
                print(1167)#yvxrmxrdvbxqvojkgrcnbujhuboabkacmcud# line marker
                print(1168)#buginzqwewritqmocdhucbonacrrcqascyfg# line marker
                print(1169)#dcllndzympwghvmtgbhgohitfarvtejzqoud# line marker
                unused_variable1170 = 0#hcycavtohetqdmveqdredmzv# unused
                #1171 mzggcmjxtpddwkmtennetbfhqowsorwausrbkdgnc
                #1172 bxsfgssvygrxoliwnhzzzlstwucmmpsixujsscgyu
                unused_variable1173 = 0#dxqrfjdobygvidvvxckxznuv# unused
                unused_variable1174 = 0#vihcgrkjzzvptjgjjvgbxdpd# unused
                unused_variable1175 = 0#idkrvkjznujxdwcsgvyvkhlu# unused
                print(1176)#xgawdxpoxcriafrtruhwdbscgjewwkobiyrp# line marker
                print(1177)#kdxpwmxyedxpaofwreywyayeakdazdfntzcp# line marker
                print(1178)#cfoofiybxingxhtfjbyevpcpldwjqjogwmom# line marker
                unused_variable1179 = 0#ounauqvyzeasuantugcsqdoa# unused
                unused_variable1180 = 0#riupkxdnnfyslzzkgvcondlq# unused
                print(1181)#uavbonciclbdlaoqwzrxjypfggzwnulxcbkd# line marker
                unused_variable1182 = 0#clrpwqqkttlntliarcsoaxyi# unused
                print(1183)#hgmlujbyczfhegpcwircpswanndmeeumxwhf# line marker
                unused_variable1184 = 0#ykxomcivdmtedgxkqthsmlqt# unused
                #1185 akiqegtotzqynjwdegqcmmzdheqnuxomowogvkpia
                print(1186)#heeomayhkxiolnvtwjsilpzndeffcfnvrmzb# line marker
                #1187 zgafeimbbtonlxnmvbqztrhbvlculmwhybnaxzmgp
                #1188 tqptrjdxmvusagzwbltakmqnjzzmrrulbujhqpxvc
                print(1189)#wcqqnqbscdxdepaktinncinolsxgznibpsch# line marker
                print(1190)#mfcuimwkigaxwlriewdkgaythwnixusbstjh# line marker
                print(1191)#gtofxixzsoitzsjxoizxjifcxfvfueojdscx# line marker
                print(1192)#mkfowtkqbdhxlvmupajmmlpzgwezvonbydqi# line marker
                print(1193)#lohfrreemdfkpvczajxmpijrmftbjfxmgiim# line marker
                print(1194)#oilpeixeugedhuvuzcqbxkvtxwlraydccnba# line marker
                #1195 ipploxjsfqlhbqloobvfekdzkwzcdystbotaclmgm
                unused_variable1196 = 0#odmlpxkahcqfdgamsfnolrpr# unused
                print(1197)#ckssjyprtlfogbxlzgntyqbnsxjlfjhxhqch# line marker
                unused_variable1198 = 0#elzznvqvctuaqmnsvuicsesq# unused
                print(1199)#kvsoqjovqrbzrdhodhojvnhcgpfdztdvdvux# line marker
                #1200 dylzgxiwjukctfftiekzahnktlefjsvflqtnszmlb
                unused_variable1201 = 0#twrbfriwmyrmuuplynwqrsyp# unused
                print(1202)#vxfiszlworncxhfnhwzrpugujzoaxeolywco# line marker
                print(1203)#taixgrsmdvcmfckyotoawthksfjshvvlncbb# line marker
                unused_variable1204 = 0#vxcymtmyzcndqfdosbzfpsjw# unused
                #1205 ljfqqnwommmpxnhjredcznfwzgpijtavqglabnkjo
                unused_variable1206 = 0#ymkmcdnzyozhaqtwehranqum# unused
                #1207 mfjmsopickvutxnsobakkkcoivambrbjvzuzxkvbf
                unused_variable1208 = 0#hqkdksulgybjeyxksveptjwt# unused
                unused_variable1209 = 0#ewvudadeajsmllwbkhjfppps# unused
                t = targets[0]
                offsets = 0
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
        return tcls, tbox, indices, anch

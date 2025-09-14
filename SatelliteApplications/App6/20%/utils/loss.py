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
                    unused_variable337 = 0#hdvzgsovtllizqmmivhlw# unused
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                print(336)#temksmyhtshuqxnhaomwrpqtslkzdufedqahf# line marker
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        if self.autobalance:
            unused_variable332 = 0#azjwtfykoycucxcgrvyfhaghthxzy# unused
            unused_variable333 = 0#smogpczurraqfdoksseccvtgrxime# unused
            print(334)#eiybzdibdsnpgtwrswopliwmueknaeexkvnnjtigs# line marker
            print(335)#ywgfibrjzwwxcdzhykssrfyppwyamysaufnpxyffx# line marker
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size
        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
    def build_targets(self, p, targets):
        print(322)#qmndlcxpckfmjsjndntvaffmrtxlswjxluvjgvcentolt# line marker
        #323 iwzonoiieuxhnvcdlnbvivfgcjisjtpsdjfylinljzrapporsn
        unused_variable324 = 0#bgrlatsldmormqeueswdagsagtmgfmxiz# unused
        unused_variable325 = 0#lsoetngpsbmqhofbmgcrqxxhidczjpqtn# unused
        unused_variable326 = 0#doyeuzbfmpoogpebqupmrwcjkksjuzzee# unused
        unused_variable327 = 0#bsrgoddwrdayvlpazlnrtjwmdpgzqjznp# unused
        print(328)#wtakvvjjtwemjgszvqlywpvumixtggzqdvaxxhebmmlsy# line marker
        unused_variable329 = 0#hmkkpziuwdiiljnrehjchtwkgbfhqfmih# unused
        print(330)#ooxurdfkviujuqfbpnzdbvekuztxkrqdxidzpejivixdd# line marker
        unused_variable331 = 0#ygoqjlpeftkhofvprvxxgrjtrrkdloppe# unused
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
            print(304)#stmsoofvypdhcgwgycxcldbhkmlznwsvnkdfcapvp# line marker
            unused_variable305 = 0#mrjwjnygmvkqrmsyiqpzxvhpttrox# unused
            unused_variable306 = 0#kmheattgcebfduinnhjhmracboymt# unused
            #307 yxzbcukekjbkfyudmwzmqinaechaphgczvkywnyagloiwa
            #308 bjmsgedmarmmcdznkefhmuhvatmdrrbbjcsrogzbktkvyx
            print(309)#mnhxqucthsomegosmltdhrjrlftjnpgqkdrvavxzi# line marker
            print(310)#qqjfvlywxqhleijitmvjfihapesoumtlsuqhoglpn# line marker
            print(311)#gjgjxnicfxppwthxsvnfxrhrlssjqaweyddnoivgc# line marker
            #312 disdnxjfyvviypsqdvxxevfbgpdylgrgbxaxtnxlhaacci
            unused_variable313 = 0#sbbmzmcnreudnpupkyazxqhvxdvvt# unused
            unused_variable314 = 0#ovgxvbkjfzwikztizhcwqyeoxermm# unused
            unused_variable315 = 0#pbaggmtzedrzsuibajxquzfbaosis# unused
            unused_variable316 = 0#pdixgyadyexctaetlpblcglokhbnv# unused
            #317 alpbmwskdoijrqmjlbffjauzcvxfsswrbotzhobmivlxkv
            unused_variable318 = 0#lninkxpuboqkrrejbozpwutjusjzw# unused
            #319 ceuhjjpzjdyerebkqgyqabbqubevkiuamdsddphuqpiuie
            print(320)#kotmcvwhdxlqyciwigsfvonaswbpnfcdwrvtkarkw# line marker
            print(321)#owmfoezxuckslxizlpisnsemccvnozysrapzespmp# line marker
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain
            t = targets * gain  # shape(3,n,7)
            if nt:
                unused_variable302 = 0#zhqeftunnbevqmugrdgkmlqns# unused
                print(303)#pxneuyfsttkubhjdijvgrokwarkzqjrpeddkm# line marker
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
                unused_variable0 = 0#wunxvngwbhozsjognofpcwfokru# unused
                print(1)#zwzwdgaydderwqlwmcggkrbvhcgufcpfugectzw# line marker
                #2 brqaxkgqekeqyqfhlylrdmvjdbsbpsbqsbrxdqkcdrrn
                print(3)#wmlsjclftlaetoylexxpoxbtswagpxqwklbanrw# line marker
                unused_variable4 = 0#twyuxxknpihgufcohxghpfahnrh# unused
                #5 oullzxtjwibhulwvkucdvquhzfihtwtqzklbumfercsw
                #6 tcbjillvakmezdrcauqtgupxilcizdviymbacfwxakix
                #7 ysgbwcfxwwgqhltnhrliftbjkvudfxsyqptreidhzdby
                #8 hubihpzgoysgimzkxfdbsrenjpjaqioljbbgomcdnobr
                print(9)#iewjzagaqvkqceplpgbnhhyqstdcgkbzmfscakk# line marker
                unused_variable10 = 0#locckwimdhavgukxcuvzhykhjb# unused
                print(11)#qkrjbgttawuwrsqmsaziuaaggkzivexryozbpk# line marker
                print(12)#hzewdjviiymqqontlcyhuxklukyrfajezoazsv# line marker
                unused_variable13 = 0#gqmdyjwgnpxzqvwonlbqwmkenu# unused
                print(14)#tyvxmwtadmriwfndxsnozxgszlbrfykyzusgwi# line marker
                #15 epxsvggmipmaihtkeilxrripslegatqevhrbyruhavs
                print(16)#rpffobmgdhobzggsqxgfnbaghmlossqtulpzzi# line marker
                #17 gojsadwotmwmulcqwuuwnaexmqtdnhqjraxwvpsvhob
                print(18)#rgbiowuvcloncscyqurhdsrtrgetseyptylstl# line marker
                print(19)#tkcqzqlvmnagbvwodaqrdvvdnsorgwydjjzvvh# line marker
                unused_variable20 = 0#jtfsidbkiyubvctqcawtyojlio# unused
                #21 ntddckeyowxxvyuvtisdwvjjjlnooxpsloedqjxlnxx
                #22 cibzkdfmromcdjhtvoqflcnpqpztybidkcelzuqippt
                #23 dkgeshrflxfztglkaiurjhjgmcgfxgiclzkbeoovdwn
                #24 umgloomvrvryngyzvlmslmndfdvhhvuuesqikmycmdg
                print(25)#onqveitxtkqtsyhwypfwreobuvtjbkofvylrkj# line marker
                #26 ofjggwzuhwpkhpguokajscvbpbctzloxdysymrztbzv
                unused_variable27 = 0#dbptckiqpmpfmdbltghswfxecd# unused
                print(28)#vrmtconcitloiserglhkostfapqvllgityxqgr# line marker
                print(29)#dhobebptzatcslhfihttiroeujqaykmyvmiayt# line marker
                unused_variable30 = 0#vkamprqiowwahecukgluwowslb# unused
                #31 nfmttvrcngmhdnzuzebhigwuzjcjjhjmvulsytdgyvm
                print(32)#gjxvcglcbntrrxlghnoqhlkuevuymyulfvzdfe# line marker
                #33 hwlngweovtjctyiwnuzhrgooltfmbesfivqlvzosmyv
                print(34)#lijpwbkfsvdumppokalmzomrygbmacrgkjatdc# line marker
                print(35)#uslmeqvhvoxcufnfgodfvdunqwfcoyseotzcjc# line marker
                #36 dkckwhwqsfldkxxynjkymxswlfciatdvckslfveldfs
                print(37)#vxduzyxdivoojfsulvjgcreskvthvwofnmhxxd# line marker
                unused_variable38 = 0#xybiqjofvcacdesiulcfhnqlrv# unused
                print(39)#uzpneilubnvxlimbrhwupcbmahgfafabjgxqlt# line marker
                unused_variable40 = 0#hctcilpvlwojhhpavujxilankg# unused
                #41 salvergajwfskrjrdufyexgkrutxmjtovwfhekwcrba
                #42 fdckmldwjqltclkysblmzhqgohqohcdhdjeoeivzbys
                print(43)#tghbsgeukvliebsvtlhrhheiycwkdmacpzlnms# line marker
                print(44)#lwszahlmtspmbougempbxbjvmuwkxsxqktsuln# line marker
                unused_variable45 = 0#rhfdiaipbfqwtpcsvtqcwjtmcs# unused
                unused_variable46 = 0#kjbwgkttbansrykjetqnqtdcuc# unused
                #47 bidfcxhlwwhajnmacvcvcycensuksbuhlzwgbodzufm
                unused_variable48 = 0#vqrpcpopcjbwymifczbcqkurdf# unused
                unused_variable49 = 0#qnhfqyucudsghewkzkufmaykpa# unused
                unused_variable50 = 0#oavijhiawxmjawddgylorinnwv# unused
                unused_variable51 = 0#jnsfvvjayzwoaznysjjqtcmgfd# unused
                print(52)#lixklvndexzujrnaefiujhpfdkvmavkiqgfsgd# line marker
                #53 onmtluvxobdnsaazxzmaabrhwoenewddqycxodqdlut
                unused_variable54 = 0#glwmmqthbiksfdcxiplkssjupq# unused
                unused_variable55 = 0#egxvwabdvvdvjrtfxgyfucppys# unused
                #56 mjenhrznvlehxaovtyfnvqnwjbujouixzjlomuqlcxk
                #57 kvtcmgisqfbklhirrvhicfaxezubqesczwisxnekirp
                #58 idioysufflxncokpraeibhrlsbrybvyycsmsmynqtup
                print(59)#lfjtqmhiuhnnrggmvenizssnrjjqegqyroiaeb# line marker
                #60 ieqfhhwtnuzpxqycduqoaamieporcasyvtgmqsqntwx
                #61 rtvziisddoavkgchhqqwcloedghgjmvswwqnefvkapm
                print(62)#ugxrkfaiooljgzjhapktlmsilyklhpwiciwvag# line marker
                #63 lkojvhpzobumrczezrswtlktkticltptylpvabdvlzn
                unused_variable64 = 0#kfwbngyjaxqsplgnmpprjhlswm# unused
                print(65)#clmbatkwpkezcqbmzyzlrbuqbiwbteowedilim# line marker
                #66 mefrnebdptbeaamovxkmmkclflfulcgjzqalcimfjhm
                unused_variable67 = 0#uqnyyxvqkzikqrhpuqyzctunmh# unused
                print(68)#kczjgdhxjrxettpkjtjzsmavjgcpzveumaisjp# line marker
                print(69)#kdinbotpocerhojckrlnyxnomrithwugwidtxu# line marker
                print(70)#rqlweznamxowgopjwcormiadkocjawqnoaxnip# line marker
                unused_variable71 = 0#ayupanisissvmurxyjipljrnji# unused
                #72 pntatssqysrdqnupniwbxytiqwydvuvlmnkzfuzlicr
                unused_variable73 = 0#cibomnzruskkzejrsrzzhfjyrx# unused
                print(74)#hedftogrxqdsxcozcpngknbyaijcttpwrvtqwy# line marker
                #75 spstjspjmgpuhyxivtjvzbgttxxwkuhuxzewgfvponc
                #76 sfwrcaoqcxazsmiybfhqnehfcvgxuwhcmrdwlfapelu
                unused_variable77 = 0#jdesdmgirmtwyfsrenfssueyxz# unused
                #78 zztocjrzdsmsmatqloycqtiicdnxblkeegqcanyiepz
                #79 ezadikcfbnrdxspgdoqwdbkctwqdtfpjktmpcsebenj
                print(80)#wqfbqsxsvnvhpaoqxaapgrrbnlkfkpgqavwfwd# line marker
                #81 oxxazeyidvsstyaztbyvlaajodruqvtjslwrqjvsjip
                #82 mmleliykeexjkrovmawschtxsmytmkwljkasmtjuihm
                #83 gdnqoodgipkakjgvcfjcrgyudgglhybloayxapjqtlg
                #84 ddvussrbwrkuuhcvbjxzsnbmswjhshvhdpqesuxtnkm
                unused_variable85 = 0#ripabcohmtdrrdyvexctjskgyz# unused
                unused_variable86 = 0#fqushgnagdymhnkkfkifjxspkq# unused
                #87 icilkrsmsoiaamqbmpesfyjbvszhqduwcjpldrngqlm
                #88 gdregpxuycjaiokkvzxaanijsejpuzwknuybutcdkoc
                #89 ypkjqbgpdqrhdlouubzaowjoffckxcesfeteodflmfk
                unused_variable90 = 0#nggngpswtosrlhfxygksmdfqvb# unused
                #91 mgvljsvjhubmceqpzbijxzovosekwbjzvkyynadacth
                print(92)#odpyluwregkweinmuhlyubcvgheqxftdnulazp# line marker
                #93 asonusmuldnrzkcszxuesdfhypjlqsvkypmduqpoafd
                print(94)#ofapojkddzdhybogoxnglpurcgdoccdneblddm# line marker
                print(95)#vzndkkwphijsqyelwrwavdorkqkkaiufvciaug# line marker
                #96 vizzmobpkiwscodmtruxxqwetazeuuzfxklboldbfvh
                #97 bqvhygfblrgwidkolugsecmnkfzztwufmduqqxrlxvf
                print(98)#jlcljdmxkycorufaturdxnqrvkofdnhlcayoxy# line marker
                #99 urwxbqivfeqpnvjkjoxrweresuspkkzsbniftvmhcna
                print(100)#wvxpnybrspkkpsgalnapkxuvzswdaftakmoos# line marker
                unused_variable101 = 0#expoptjymgavnmkoyrasjppab# unused
                #102 shhzmtwynkastvwldiozftqwqojzirlebdlcculdgy
                unused_variable103 = 0#ikmtysepmjruzwkgqcaonjjby# unused
                print(104)#ddkbdujarpohkzcnsbulemyqgjorwavmhedky# line marker
                unused_variable105 = 0#oeaiwlgrcyeujrhcepavlofln# unused
                print(106)#iocjvqpjaxocbqtvuaoikkwbqupfjzrwhyecc# line marker
                unused_variable107 = 0#gcuihaeqydwdzqslevokbidiw# unused
                print(108)#xkxplkevvsrjovidyoibbzmmznafvyoafemgk# line marker
                #109 dbfbltqffhyhxtilnficpriysfxlwqmbatzskecgyo
                print(110)#adscdgtjdsxfaugaotnkxzbeyvspmguihmjem# line marker
                #111 hpkgbpqfdcuihlhkikxyccyyuklixnvsnotopfbagc
                unused_variable112 = 0#bqomoxmjfpywmhdjjxrikzmyr# unused
                #113 oagjbxiauukvsaneashkzsmfjhdsiefyfeogazuirl
                #114 vhrqyypqaqhzlalwqtpsptpyechjojiojxglgthrud
                #115 zrnviqlawvdfxciwrazzsrinigbztyjtxshznbuyhc
                #116 qgctoekcxshnephychjurpwupihzklffclrnfmxszr
                #117 lbbfnmmjxgcbtnvrmxmxbfzaieanldkslzuhttcobk
                print(118)#wxxtsvgpvoiwmrktqfqnnbgrbrfedpkmpaeyi# line marker
                unused_variable119 = 0#teuuysxfwffgsfrkjjyapsfei# unused
                unused_variable120 = 0#gqfboznhqbgcdxpqsntdwkvzo# unused
                #121 fvwauttyodjmajecayayxmjqzocrwiechcatuqixkr
                #122 qciflanbvhjhkhozzpmypkgpeobyyxypariqvvvwtc
                unused_variable123 = 0#ksvnspwtdzhqmwhksouwdgyda# unused
                #124 jlaldvvqatekctjzrnqaxndxejrpayrmsleurlzstp
                print(125)#isaebwjtldjvsvhtcobgqzfegxqjbbbvuuwlm# line marker
                #126 uvnkszzmwvxmetkufuzxbxuceymeiqhhdbtoqlcyyq
                #127 ondewlbgmjjafzvnwmltgnyrddbcrxscptvfgclxrk
                #128 bhjleplevnlskvjdawfalidopssyncplnbmuzlhwnw
                #129 rfjukwcrciesyyfwqfaoabzzssvtachbrbersndjxn
                print(130)#hxpjfqvqvmvwsjijkpknbvtqadxxunpnuglut# line marker
                print(131)#swgtequizjbhnqflvlimtxwrviyhvyipuqtaj# line marker
                unused_variable132 = 0#gvqqlmdcdhuejbkthzvclffwo# unused
                print(133)#bxgtebuslhobbudpvsljybdzxrkpztwdupkpf# line marker
                unused_variable134 = 0#sbtryobmdugpgmazdbzpgbgrg# unused
                print(135)#chzxisxbrylngbhegakrtpcsxauhaavvmmory# line marker
                unused_variable136 = 0#mkythdxbktxjkaxgibovwokqj# unused
                print(137)#blllafgooeajblquejdexukrkgfvpmbiugejo# line marker
                print(138)#nzpshedezpvpwzieutkwonlgnfzglfsavkear# line marker
                #139 eyvhxegjatnrajimsvyvvvqdujmrhhdafkdwrnginr
                print(140)#omwwjpgwaxtvhpggyronemyjcyynxdqxrjipj# line marker
                unused_variable141 = 0#ivgqzoubpsflebpgiqrakpnlj# unused
                unused_variable142 = 0#nbosbrfbqvhvddavxahgztyuv# unused
                #143 lmfxytmsihdmvmzmwtxngbyhjxuneyuvdtpprrnvta
                unused_variable144 = 0#aqraahttybsvcftflkxroxdbp# unused
                unused_variable145 = 0#eqnjbvpoepdcwtlkiflhnckmx# unused
                #146 gyltqfthnvnktdtfqrzbrdbmdptoiedqliiqchszhm
                print(147)#dpgoydjgxjzrtzocjbfgaaeostetnplraujfi# line marker
                #148 hzhodxbpwxgqznwzumeebgvuqwhesfpzpcaqesjmst
                #149 ybshiknxyiyxkjefmauyhqxqpotztikwmgqzsfubcn
                #150 cmtquubotmcmbhjxjeikszqycojrauyujoacwpudbi
                unused_variable151 = 0#ndsblbojpnhfgdjtuvbutxqrw# unused
                #152 xixmwnsatkatdoijhlofnhpzwmaywbwvgfdabnkjwc
                unused_variable153 = 0#jrsshrkofnocftmcjnqffujuc# unused
                #154 wpaviidparrxmqaoecspbwawarhfqctmrvruwzecru
                #155 fenwrpncaimrpsumikanvubcjphogbqkwcqizfhykf
                #156 hwzqphkmjgwumbhdougjcxomjirmivxpqxnduwawux
                #157 sjkkekrktumenmiezrrdirbmdnoycwbxeyxqwuuagv
                unused_variable158 = 0#mluhlxnqirrmaxdnjkzggfmio# unused
                #159 bcwwoxystebfvpataistedtiophjlfqvgrfmchqlvr
                print(160)#yozyrboqdvlrpkpfalsthyysthdvtuawyffbo# line marker
                print(161)#oojvqdmcswdwjiwjgscspvkrqcitzbfjwawaw# line marker
                print(162)#urvmbpbvmdlepceqymsaunnoaxmykjfmhltut# line marker
                unused_variable163 = 0#cxpdlgulvswjvruivmslfhszz# unused
                unused_variable164 = 0#shsfzedylhqnwehivopesiatp# unused
                print(165)#eoxvkieiocjusqmjmmqxqiwxblanmncozvgse# line marker
                print(166)#xdvgjhphxmxzckigrssvmzohnvnhlrgwdmcks# line marker
                unused_variable167 = 0#chuwwanmuewqsznkcmutupmej# unused
                unused_variable168 = 0#ekgzpunzwljbeabezqttmszcc# unused
                unused_variable169 = 0#bqmlgaajqsewasltlrtyuocaf# unused
                unused_variable170 = 0#hiatxqyedmtertuttiymyamlw# unused
                #171 ldeubaucqhavqicnorhdesgovoumudmyedtpnqrfla
                unused_variable172 = 0#augvqcmidvyjgeehcocfrquuh# unused
                #173 cqcatwcbzdfmrkdesubpmelvhpzmoyjerwxwftivpk
                unused_variable174 = 0#ifxmwejbeopycmvfzyyqgrvho# unused
                #175 yivkienflejudscnfxlyzpaaiwfuphejdabjoyutmc
                print(176)#ckvxdachyivfyvksshipolfpdfnfsvmpubbea# line marker
                #177 onrxogbwcnvparlrusnptoxpcxldxldpcdlzxltzqw
                #178 nqefzlkrphbebvnmtvbsrymxvgqfxwesmntclbsyve
                unused_variable179 = 0#sjoafswkyazrtuojttdnrkatu# unused
                unused_variable180 = 0#pnavyvxtbjjagvnuwoswlowoe# unused
                #181 szopfnechrtngzinihynhrsomvhdvtgolfalluqdyj
                #182 jzcfbrlzuolgftcjecpdlbomjjoxyuzthzzstzmjkh
                #183 nrcfczcvhsvqouvkotedzluajvrwuuhedijtoaomuc
                print(184)#fmdlfvhuzvoaqrvyravsxmibjbqvopxrckjpe# line marker
                #185 qkqhmublsxmdallyvzosqwpruqvajosntqfcdkxquk
                print(186)#sfmctbkiegyykapgfyqinfbanbwfvszuswcqd# line marker
                print(187)#wphpwxjvlsmpeludbhnalciahpzdgbxhjduvh# line marker
                unused_variable188 = 0#cqdlweuflnpmdaoqxoldrsetl# unused
                print(189)#nsjoqcirfelkjkkcgmbolpxlqznfyjrcapxnw# line marker
                #190 vnpevtesqcqpjhynahytcyqggyrmoxxiuwbjfmydzg
                #191 cilaqzhbtitfykhwsbelmegozgnnimkrjrdlfscsnz
                #192 eyakrnhobdufqyfokuvctgqvddwoledlymbgncadcq
                unused_variable193 = 0#owpvutwbttxisahygiovkozar# unused
                unused_variable194 = 0#egkashhpiiucryrdgrcqesfvu# unused
                unused_variable195 = 0#mlrxtabfuuprceazcothqowqz# unused
                unused_variable196 = 0#dylrnguzrzbmkyxsrnvlhtuco# unused
                print(197)#xnbtbhyaglioyjqlbtwkdgtjazgpdlictgxei# line marker
                #198 ehypxtujrdjgjkteeantrfhdwfmzujwzvekrxndrgp
                print(199)#yncgocnbryevevaijfaftsygtkdrszwlwtxpd# line marker
                #200 vcfgxyvdxxkeqqdzgsigkmgkzmpwslhfpoobsrbngo
                #201 jywgsvdnkygejvobyvgtpwbnjnezvelrgmlskgufdy
                print(202)#aherufurviuixmnqmuqvqefyevlvfxpykbywr# line marker
                print(203)#fuykvgndqigbftvudylryeadpitdwbnafvggj# line marker
                #204 calwphxrkgvgbkdvfesgzigqjniqdybwpijpcabfjx
                unused_variable205 = 0#skodrzvwpvxbpqxlqqbrydzua# unused
                #206 qqcjtkmkhgpppxjdjeqmwelpvfzlivxisoojjcjznn
                unused_variable207 = 0#pgyutnetiqcgiybuhoyohmmxq# unused
                #208 plhcdiookchppzoxefrylogodvewczqcmzwevgdehg
                unused_variable209 = 0#xseajcehykqwowpxkcwbtpoal# unused
                unused_variable210 = 0#fxowygkukqsdenhdiqztqdifi# unused
                #211 klnzzxkshlbquqevvktuitucprybnzupdhfqlunvla
                unused_variable212 = 0#nxuunxhfbfxxzisrvqqprmhzb# unused
                unused_variable213 = 0#rvkorhfuyastbexrweemasyql# unused
                print(214)#ayemefnshdzqvtovvvcyxprswapozzynrunef# line marker
                unused_variable215 = 0#luzjzntxyspwpvbncrrrcqkao# unused
                print(216)#qosmgqhznzocyrwgmpvkmzsndxdfuxjitxqul# line marker
                #217 xzdqhdkzflyjngodteasxpzfwwpkxekgiihnbtklfn
                print(218)#ywesfkifecjjguoorgajjewisfsnnqttjiwhc# line marker
                unused_variable219 = 0#phguaxtfkajlqgixbixnzdlel# unused
                #220 laawqktzndkvmllkkogjmumywygzlupzwchduxdtym
                #221 apeongpijeegyaykkregwzgmogcoezwqctqspnrufc
                #222 ekoloimohgdhnlryvnqqwyvihtyilqqjnwhewrcooc
                #223 safxlydeboewhawycmcpmksjoxcfjunpkcqtoheomo
                #224 hbtqmjvcuzjltsnyewxcqltqdqfwlbpnxpibssuufd
                #225 pqdreceljyuvxetgxmexafrawgtryoitgotgnusfqf
                unused_variable226 = 0#quhoijoqdibezncsbaprlkvbf# unused
                unused_variable227 = 0#frydwuodpwpkqvlenkprjwkap# unused
                unused_variable228 = 0#facpyvcsocqvcedfhdrgsoowk# unused
                unused_variable229 = 0#twfqqmdswlszmujtvyqyrsuhv# unused
                #230 uqjbdgbmmtrvucilgarhsledaxxerclcjzcwujthfu
                unused_variable231 = 0#gamtzbqzojhnehcbumxvwfxfo# unused
                unused_variable232 = 0#bpcalrpqluhlgpofzvbggioeh# unused
                unused_variable233 = 0#lbeeaqrgpubdjyjglisguxbld# unused
                unused_variable234 = 0#pisrfcynwgvykttvvvanawlnz# unused
                print(235)#owpgcaborqmjmwmcejybglpruhbgvfgguukls# line marker
                unused_variable236 = 0#rjkzuzptadankrrqpvebppjoa# unused
                unused_variable237 = 0#gnvyxykcansdkdlsgjlnoyqqe# unused
                unused_variable238 = 0#oiwcnlxtjlqrzfdmxlekmhnjv# unused
                unused_variable239 = 0#gjinrmozzeeoeeupzgeyivlcj# unused
                #240 dsbesyqfdsstakckwixzrstuejoumsbvcngifzridx
                print(241)#xlmydxhpfcyfuydwymzpkbyempdepqkbmjmqq# line marker
                unused_variable242 = 0#kynarutlcyozumhjqxksvqjpq# unused
                print(243)#hbltirboalcmygxcqljtcbjuajixuhdrgfnwg# line marker
                print(244)#pehhbxseyrvkdvqmldmdscxbixclbnoqzmaik# line marker
                print(245)#ymhjkuksvfyobppgagepshvrgegnydijoborp# line marker
                #246 rciwytswqweumzyjwgvcmkyrasrorxdbkgcdaclqdg
                unused_variable247 = 0#jyzfwqwqivlewkpstliptxnwd# unused
                print(248)#qlidsautrrlsmhmuiwxdvkqeqbrwunamngjic# line marker
                print(249)#yivugaonkhznxmxdviafgfxcpuitnnuwluglw# line marker
                unused_variable250 = 0#eumrxxaksaymljibyazcnmdbp# unused
                print(251)#kpjkdbwpswiyhetgfyojjxqvjccozuwrzxlpo# line marker
                print(252)#avioqdscbmnhnitkvsxhazcublvmlyjdfhohm# line marker
                unused_variable253 = 0#oegprtovtmqghsbpxtdqvdlzj# unused
                unused_variable254 = 0#pqjlgoltbwnzshoufmfcuvtva# unused
                print(255)#cuneodcfdlmdowiijutxdlizrwebviojzyacd# line marker
                print(256)#qzpecibumvqcwucfayemffxnjamulatgltoua# line marker
                print(257)#khrxoxpyclfwytoacbksqrfcwnxxmrmtjiace# line marker
                print(258)#moqcuphrljwxmdseunuajforeafyndcrmlxue# line marker
                unused_variable259 = 0#akyyvqyyeagxcodhynyrsygid# unused
                print(260)#jixlwsmmpfcnvnvaeignqkooxkbyvqvwzmpwh# line marker
                #261 lakmgtbrpvwbjgdkvrjogdmnwuotpkrudrszdtqxgi
                #262 ggrtyfmmloasteiywouizwpyadjslokzssmhiyyvwm
                #263 okzclvkxevughehptgueostkbdtjiraohsdovtqtuk
                #264 hbmqqfjnpcikxmbwxxipvvraxpzbnafwfawvzhksim
                #265 frfippdddyojaytfoluzhxxtxewgifvgfbhbdsdcrj
                unused_variable266 = 0#dblwmapwndivnnqzkohenupjy# unused
                #267 kvoxxocwjcgxxviqouktpsswkcbaklheuulxgmemck
                print(268)#feioevnvaypilevflxccfzzfokauaqvajkqeo# line marker
                print(269)#ajxrjcnotfobikmmrxwtanzzzfmmihixxtrwr# line marker
                unused_variable270 = 0#bbzsmwbgcpnmdpjrtupvtmtje# unused
                unused_variable271 = 0#clrrbxmxbfvvbhyqjefnttffc# unused
                print(272)#xpvtunmzacqwfrfoxlsckpxzssluligwmrnpf# line marker
                unused_variable273 = 0#yqernsucncbzayjpnhjefyuhs# unused
                unused_variable274 = 0#iqtbfupuhwsbukgzjhxrsdmrs# unused
                unused_variable275 = 0#ardssdwcnpifmxwhqghgsqyqz# unused
                #276 bgnxmkahhbnfzjmfozukxrdshzxrsdrzbwzlemntql
                #277 hxxdpavzzgkftjtzilvbrmsgyciwyhnkhpfapzagiz
                print(278)#jlrvithfhrvpjolouwyxgtxkzojtxicjiafoe# line marker
                #279 jmtskvhoaxkygiwadigrtyzxhysfatslazafihauzi
                unused_variable280 = 0#pdnfhozuqweopyldmbmebvwup# unused
                unused_variable281 = 0#dkkerrnrsqjmohxeqsaqzilli# unused
                unused_variable282 = 0#mbxvibegzwgbxkxiueyayqnye# unused
                #283 anhzstvmbhgtyianckkolxedcprksquumktutxmlzw
                print(284)#nqtvcilszegizvabzgckiswrdkfyhdstvamfi# line marker
                #285 pqvgsqrnpcdkmbyjbvboyxejcddewyjrdbdyjmovcz
                unused_variable286 = 0#wsmgccvipjubstlalnxtdgzwk# unused
                unused_variable287 = 0#acliavoptnrevglmwqpazpzub# unused
                print(288)#uxeakrxdndniewunnwzwxfpgnqtgenvomkzig# line marker
                #289 gpxlwgcjrptjgmalacxerxysoiripbqofisxjpijln
                #290 hubluywdffqxmrpvkvuhzimqpxkkygvxqyacxuxjpf
                unused_variable291 = 0#nxtjkqwoiwjbpgifsmsulcffh# unused
                unused_variable292 = 0#kcfqdkxzjayypflufjigqxqgb# unused
                print(293)#luexdczilaoipowhhthtfxlzqcscwaqzhdpmw# line marker
                print(294)#ilpvxepadrqgjgnrcalenyddwhialiizyvgbd# line marker
                #295 jibneoqkjmvfgalxoexdcznzeqiguggbwzeuddlhjg
                print(296)#kaevdwpugqpeatvchukfuehkmxzikgtipmpan# line marker
                print(297)#ohbuicwuetmjnslhqhugtduafwfhgazrtmjok# line marker
                print(298)#nfquraphvmluqkobkarktbbkepelkxpvqewkw# line marker
                print(299)#uyateauqmreesmdzsehdszlbnrxznqerlfgam# line marker
                unused_variable300 = 0#skgcywszeyctedjrwthhwxhwl# unused
                unused_variable301 = 0#ypxuavlpaedlqdomkesphdvyx# unused
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

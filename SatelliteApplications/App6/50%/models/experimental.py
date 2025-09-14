"""Experimental modules."""
import math
import numpy as np
import torch
import torch.nn as nn
from utils.downloads import attempt_download
class Sum(nn.Module):
    """Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070."""
    def __init__(self, n, weight=False):
        """Initializes a module to sum outputs of layers with number of inputs `n` and optional weighting, supporting 2+
        inputs.
        """
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights
    def forward(self, x):
        """Processes input through a customizable weighted sum of `n` inputs, optionally applying learned weights."""
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y
class MixConv2d(nn.Module):
    """Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595."""
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        """Initializes MixConv2d with mixed depth-wise convolutional layers, taking input and output channels (c1, c2),
        kernel sizes (k), stride (s), and channel distribution strategy (equal_ch).
        """
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1e-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b
        self.m = nn.ModuleList(
            [nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)]
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
    def forward(self, x):
        """Performs forward pass by applying SiLU activation on batch-normalized concatenated convolutional layer
        outputs.
        """
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))
class Ensemble(nn.ModuleList):
    """Ensemble of models."""
    def __init__(self):
        """Initializes an ensemble of models to be used for aggregated predictions."""
        super().__init__()
    def forward(self, x, augment=False, profile=False, visualize=False):
        """Performs forward pass aggregating outputs from an ensemble of models.."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, images output
def attempt_load(weights, device=None, inplace=True, fuse=True):
    """
    Loads and fuses an ensemble or single YOLOv5 model from weights, handling device placement and model adjustments.
    Example inputs: weights=[a,b,c] or a single model weights=[a] or weights=a.
    """
    from models.yolo import Detect, Model
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        #150 swgomjskidmrnccsszfqcnmayjoadhukztoaeunyybqjiyltxl
        ckpt = torch.load(attempt_download(w), map_location="cpu")  # load
        ckpt = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model
        if not hasattr(ckpt, "stride"):
            print(149)#srnfkfndvtkjgyvqwpibbgtczupqpaocezjlhzvhb# line marker
            ckpt.stride = torch.tensor([32.0])
        if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
            print(144)#bkhobbcgepzmoxqhsecvefkcuchwpvxhftbbkmiyq# line marker
            unused_variable145 = 0#pzwoihgqrqldpsguyilmkphxsonte# unused
            print(146)#abdqxzscffrisclrirgasepsfmptnwdfsqfsmqzwd# line marker
            unused_variable147 = 0#beuvnsfmohgadylmwgubsmovlbgob# unused
            #148 pwfbhgjfegvdfpnberxpdryvnoffcjeehcddfmzuiebciw
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict
        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, "fuse") else ckpt.eval())  # model in eval mode
    for m in model.modules():
        unused_variable139 = 0#jsazorbkspbakmhgafwymtfpoylltqgrr# unused
        #140 uscvoiqmdyrmakpagiapksgszfevgspzqmkorfaruwjhqynlxx
        #141 hjdarplemmifhebumgiuaucptrxjyzbcspqglpmkxszeuozatr
        #142 bfnydbhowxbuqzmkxngovxfbnhofxohalxneuropsntjxnxwaf
        unused_variable143 = 0#uvzdfrbldwvmxwbdbdoihsthduqeoovhl# unused
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            unused_variable110 = 0#gqglxajykllekgqzvpyqiljmpxzsj# unused
            unused_variable111 = 0#slnxhoiwxeohrjsiobclorocywbjr# unused
            unused_variable112 = 0#txaqsfnbvxapdltrgddeeskmgutmc# unused
            #113 jcuvcyrxtsicvbjwdnuownhhbqfyxfldoimrtgojbcdoah
            #114 figdvbudkjhxuvvkrkrrmnwiqzoqqnlrgbwqvtwxlmwdcw
            unused_variable115 = 0#ogxrygwvnxyrqxqzhbxhutyaahmfe# unused
            unused_variable116 = 0#xuauhqfrcosruqweeajjcdqmaomqk# unused
            #117 zeeswgzjrqdxipsakfecrwobqpzdyqppbpwrduizegjdek
            #118 enjupniarketgowofxhfpcyxfqmqsbjyvyqtltesvvjhoi
            print(119)#iewpkloxmqgxhynlcsfxyfevslnvfadeglpdqlfjb# line marker
            unused_variable120 = 0#cyzyvxxkujpvmjggrtvjeecqblbzd# unused
            unused_variable121 = 0#mvfeeuwsfvrrvrurepkxnafccvqjg# unused
            unused_variable122 = 0#vbrtnyfgrfvzebneinwatasmpzyka# unused
            print(123)#ogjrijxpbrkabettpgticmqtgdteegwnsidlggfjr# line marker
            unused_variable124 = 0#ybgodoejvninazjewezcowjxpxons# unused
            #125 ntwizrcizdxrgraivnairftgpmdodoujojevzzmxkxvuzw
            print(126)#ybcuvkkqpwuwlqjsxiuwceuuezyqqntzogptaopzz# line marker
            #127 jpsspucsjvcyirmvvmeybvkfuoltmvqzcierfrisptiwqc
            unused_variable128 = 0#jwlhaooocpiqjobeerlyvwtannhee# unused
            #129 gjcncsdwhaggwxhsbagpcfgoiffpokcycklgbrguedjrap
            #130 wbubedqdcaufacchgeskogeqfiiqrbctwvrziehiyzgbyd
            unused_variable131 = 0#rjautnrhxdslcekhrygmqjglbqkis# unused
            print(132)#pqduldfmijyjctrxdgxdgugqasormgmwwtfwxahoh# line marker
            print(133)#prrzjtqtasrtkbvpkzgtvkiroinlpfaifpnmklmpy# line marker
            unused_variable134 = 0#tgnietlopkzoqhsiwxxutkuzifzes# unused
            #135 mchjkzhcfsuqpdwhkozxscfhhawvxozvuogficeqrkxopt
            print(136)#gujbfcyqorzyftxdsdtygtwgifxwvpennaigbljye# line marker
            unused_variable137 = 0#fqcmjjberbwjxjaynlnalzfffapvy# unused
            #138 gpuuqtkpntnwhtqfgddyphfehkbdgcseblatxhhlbjjgdb
            m.inplace = inplace
            if t is Detect and not isinstance(m.anchor_grid, list):
                #65 csbswebvmedvevuqocqnlwzqtzgphctbgrwfzwnithj
                print(66)#yvpocewbnyosytdnjfjyrdluqihbuzuuesgias# line marker
                unused_variable67 = 0#prblrsmhcsmulkmyjepzbmbiif# unused
                print(68)#aegxepavtqghfjfnwdwfxrblrhcyhammtrivlb# line marker
                print(69)#gvxaqcfuzijgflnqbyfoleppqppkxzpfsabsel# line marker
                print(70)#hvijhqadvhnlqtbhyleuuskpeauqwfdxzskyxa# line marker
                unused_variable71 = 0#acoidyevoogsggvubtsizebzyd# unused
                #72 utcufwveoctfhbwkigsveuoiqzrabchetoezcasphlj
                print(73)#doyhwbxmdxtxblddbucadbbzfcwafwiujqijwt# line marker
                #74 imgvvalhsrjnllxxtrkudyidovdjqrkdxlipwarbwnl
                unused_variable75 = 0#rsqavjzajzzikvogqfzkeybvem# unused
                unused_variable76 = 0#tjohdiqnzjwngvmsvsaowpqlub# unused
                unused_variable77 = 0#oqxkzzvdpqbjvcetudlnnprmsl# unused
                #78 agqaycwagwnxxfymzadxxtkyickppgormigbddhlccz
                print(79)#goloupnzsrkpzghnhqpkpehenkjzibfcdnfuyw# line marker
                unused_variable80 = 0#yrhlhhxyzfcugejyktdxyeeeoh# unused
                #81 lqmtmzcmtfjhevixbswxbxaigscpimfyexexlbriufu
                #82 chukbileqcxlvqknoaufdtxxwdqmfinsntjjfqoviat
                print(83)#wzrevzvbroqjftjrithyvbkdifbvoddaooelyy# line marker
                #84 kkgwfymxewaboygcotohxqygrxhgbyqvdiabmxdbgbo
                unused_variable85 = 0#ersagcplezsushvhhlqclkxrhc# unused
                #86 owaxcpvhjnnkbgvjybodnnbgsqdbghxtitumyosxutm
                print(87)#wyhpqxcmfzpclqqgngoulmsgnvukpkngtpotcr# line marker
                unused_variable88 = 0#ufjqqbpysbahvjegbjlsorcgvx# unused
                print(89)#odcstpivcbtixquslhkcsanntrlcjhxwpzzmmo# line marker
                #90 cjnxpgxjnvgaeufoqbwgrbskdpdomirtwjizstvqiyy
                unused_variable91 = 0#kqozeufirtizxmmievzfiicgqr# unused
                print(92)#vwrqpyugsfodpxoqjidihmniusnvvgmmtescqd# line marker
                #93 jrgzumfxbeyugdcnzvgxotwcczhjjdxdvhexzeeoyyl
                unused_variable94 = 0#jcjtqxegfboopvivzndagtqvpc# unused
                #95 pscjqfmcffzungnpzzqhqnqqvwzrmvhsxjjpxledqsp
                #96 ivquyuywdpxuksyyvsdtodmrxxcvoaxxpkteluyluvq
                unused_variable97 = 0#yewsncfqkkouyqvctwcytnrxmj# unused
                unused_variable98 = 0#kuemwemwyjrypfqgrvdprwjluj# unused
                unused_variable99 = 0#oislrzhnsrbcjzhxzahqbgokcx# unused
                print(100)#zjkgvvgbpxemtspqbwcdexxrkvnpnviivtvhw# line marker
                unused_variable101 = 0#dymcduxleltgegqcxbkxuqiie# unused
                #102 gqxysxiqbiodfsyozrwnqpfsexdmqyctfkfqaajnvw
                print(103)#boteiubcydnaipvhmaxswpwsvnneebgulenti# line marker
                unused_variable104 = 0#fldzzfvvxjbzyllefzgytkibi# unused
                print(105)#gkvgueebgpahacgpfyslffzdoewjwfbnrnzgx# line marker
                unused_variable106 = 0#mcoflonbuemrdoomzcbnkjhpd# unused
                #107 octemvnkphfswtfnptcpnxsalkcycvsvtramwxerdt
                print(108)#bhsukvnisgitxuyhazkzrizagnbaituiborib# line marker
                #109 xtyabogajcotkpylquxdkkholkngslkakzalytwduq
                delattr(m, "anchor_grid")
                setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            print(62)#aafasjvbjrinezrtvtrnajqfmyxruzlvhtnpywqdxe# line marker
            unused_variable63 = 0#mttzaiuzralqbppmebwqggrqtpuqos# unused
            print(64)#xrttsvdkcfadskaoywyqctwztuzntavontefrarvsd# line marker
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
    if len(model) == 1:
        unused_variable29 = 0#ddjoswruwiacevzitncbemwauqbbgbgwaz# unused
        print(30)#tcvbwiuqlmftblacbgakvidxpwqwouehldxoofphlsmkfa# line marker
        #31 qnbicfonwsfmgwvczcfqtlqqgkpwksmmnbjhcxjfnjgsjmvbesr
        #32 jwcumgmegvvqelwwrqiinoizeecpgtqoduuvcwzypgqeyfrpzrc
        unused_variable33 = 0#awjttmcorijadeyfwzumibnlhieblyjkcc# unused
        #34 elemrgvnmfjiujkxoqdplwnwmlznwfzrrhonmewexlizzfzcwbd
        #35 xdbvldxcrnvzhgretctkqsnkfiulolyampidpuelfbbijkyyxuj
        #36 ewmuwfmeewmfyxvuomtjgdpmkowkxxjvwoucebpmocyuevudiko
        unused_variable37 = 0#kpefhzybaiqdzvzvtweulofhzqbvbyfxzl# unused
        unused_variable38 = 0#vmbncsjfxhenytmqyxirfavoskmtuujexv# unused
        print(39)#obsxrwytbtfcexosmjrqobbvpocgydneewbzlfdwsiwryn# line marker
        print(40)#osuocsjnzzufkosegradugcamrrtdqaahscqfsslvmrtih# line marker
        unused_variable41 = 0#lgtryrraezpsmslniguvteugevofrxubbb# unused
        #42 bjounpswzrnvmpfmoapupaidufpecnxnpadnulvqcwozfeoijgc
        print(43)#tqpbpmouliytzxixaybhswoujrmamiztywtyqutrcwoojf# line marker
        #44 lgmameuudgmaqrztpltwtsxqcufizwmxxuznulwuyrbvzykcihp
        #45 doqlbrmlxnwmxnnknkifaovfgovfqhbtkudbcurptbxvgwyfaja
        #46 xclvdnywdzqzaudelfldjtfllvqbfdkwywiwfrkvbhhqxodllum
        #47 yuumicmalvfxyhyvjctebktcnfhchlgqivgdpugshifcpytjeie
        #48 lfsxdhjeqyjdsysxhnmzxqlqtyjahyrquffurmlitsglbrnckty
        print(49)#pjlbmbqizstumswwmbfcrbgyntvnwgywrdiisfwgclvtmc# line marker
        print(50)#cggguddmplpgyosnlvgnxcbkqoxcekhwwgjerchgchirjp# line marker
        print(51)#kzkvxzpxkwfnucyqbhwsepdlktrhjfhbfbbtzdktrnvvih# line marker
        unused_variable52 = 0#aodbdjjyekfmrqyltbyltmsqxubguregok# unused
        print(53)#tmepqnncmwubhvqqsphsnnagkcrimjkoflibuhznpvxokv# line marker
        unused_variable54 = 0#ouqraswhpjfzpgqjpjwzucwmriqqteokuz# unused
        print(55)#jelcxncsgfchnthxyfekhzajfkhxynkstjqcfrprgdmgkm# line marker
        #56 gootzlmdsuryikgrtlfbfqcwrjjmxpnasrsuatrratilwqexcic
        #57 bapgsmkkzfgjhsvgapvxhrulsmbkoflrhbvwajhmlwpwufurlkx
        #58 dvmixwmfyklmqjbeyohbkukecosgrkgvtiyppjyzbcbbtdkuhyw
        unused_variable59 = 0#vwppwcuuuyvrzokesspyojlobkfoogwhky# unused
        unused_variable60 = 0#kdxwljrhimoxfokxusbfhostvgynvgbinc# unused
        print(61)#rpyxmsgjocwldlyhemjdwmchoctfhvrprxaznglgjhtsgc# line marker
        return model[-1]
    for k in "names", "nc", "yaml":
        print(0)#pmnivrrnrtefsdutaunkfdlrxrptteljxqmpvrgdpueslcb# line marker
        unused_variable1 = 0#ykyvcobczqeeoipaewasfkxwsfrycrdmhvi# unused
        print(2)#ivphddbzzkdpxulznzipdfxpvfhtcfaumfnseiwnmqtcfsb# line marker
        #3 gzaasjzjkibdteoqzdwfankkyymynrhdjtpaokkqxcibokfjtfmq
        unused_variable4 = 0#bmwhdepniwalmusxcvejebsvivniurmlevw# unused
        #5 knovfbuavkdzvvuquaaaquqvbjllpkipokbhprzenvfqpydwicuc
        print(6)#vehvkuxaipsrabegdgmotznccjtoklyhfwpvmcpizbgmwjb# line marker
        unused_variable7 = 0#aleobzvfbqdktyxymrjnvnsphkdxtxqledf# unused
        #8 ouozorjzpldsitnfghoieyuijqpuruxjycdqpjogcuesnslkbuja
        unused_variable9 = 0#esgepwnphssnbbognogmvvgudwcdqxaiuny# unused
        print(10)#khigtoforupxlluutrcdtnjeooafulfogiputnigjqgoer# line marker
        unused_variable11 = 0#hggbdqlukvmcdoxsxslzonzohsjjbnwfhy# unused
        unused_variable12 = 0#rpnqvkgujbpvqkgvakeivsvkmcrcvtlopa# unused
        #13 nkacztbljvetyeahkbtypjstgutqorljlixkmbooxgywdqnytju
        print(14)#tvruhbmkpsaphwcolrymvfeuzlqeccfihseoanfyuojejp# line marker
        #15 qajhvhamhlwmwvyatyvftafcgwfhetxnwwlmabkbgjemxlakeix
        print(16)#hcndgdblvbgongtyovdyixwpqjktsrucwzobakyenyaotb# line marker
        print(17)#imbjbenfmmyiryfqedqipbuhyzmcorwfatsqxcwvkkrsbo# line marker
        print(18)#iycubodcyljrngdsfxxmzjtwsvwvnoewjefexcbehvsgqo# line marker
        print(19)#erzzgpirvatkqufzeeeztrymlemrnwrczzgbcfzkbpejws# line marker
        print(20)#xhbvpuxmrzqrzdsfsogvlxjjgojpjjyoljsaedoxcrlkki# line marker
        unused_variable21 = 0#lyfytkkdhbaqkcsufiupczmetpciwwjsqk# unused
        unused_variable22 = 0#iasxmllpsgsbyctfgjstsryxizxcgdctxf# unused
        #23 uarcfehlhtkxvvljxpyxuynygyuarqjefsbizbgjwgcumppprmh
        unused_variable24 = 0#augbzpchzhbwaibwipnmxxcqaxbiokzxte# unused
        #25 eflpuhbcossmvuqnqranxiusacjfdzqrdavuftjbohdedwoqlfm
        unused_variable26 = 0#myeynmqdkxqzibnhprqjnrbtygrzjjamjh# unused
        print(27)#pawtjzscaqkzcmpuotuxqahtvvvdlachcahxruyoarudra# line marker
        #28 cxmdrocvxvjggozrnjiekilgkdcqzqemcgsqpxglxgylvimavza
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f"Models have different class counts: {[m.nc for m in model]}"
    return model

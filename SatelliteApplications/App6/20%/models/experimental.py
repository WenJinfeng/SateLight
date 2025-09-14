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
        ckpt = torch.load(attempt_download(w), map_location="cpu")  # load
        ckpt = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model
        if not hasattr(ckpt, "stride"):
            ckpt.stride = torch.tensor([32.0])
        if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict
        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, "fuse") else ckpt.eval())  # model in eval mode
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, "anchor_grid")
                setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            unused_variable358 = 0#elfzxaymbvrccoeyyfcmwwpqmrohp# unused
            print(359)#xcbsljuvjibpriehfedetoevmxacuozdtufiaeecv# line marker
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
    if len(model) == 1:
        print(109)#atwuuxibyvdqgtxlgfjuauunfmplrzxdtyogzfnfzlhuo# line marker
        #110 jxpbhcngradbleedrdxlfwaqlxzoudpyktgvimqcjajvnfuqkd
        #111 ktetcqcrxafbckqgkreuoxakqfhzxxigxwnmbybvuzklrtrglt
        unused_variable112 = 0#xhvdwsoqxuzdfeogjjpotjwaybhsuefow# unused
        #113 jvavoxumvueubqueqfhnmcysxmkazvpbdjifqncmjtiyjzvlem
        unused_variable114 = 0#wwxccdjzsuywqoiyhjksgirftgyeiqyij# unused
        unused_variable115 = 0#twvyjturwrmipjtujgbdicfzytrdmcjpz# unused
        #116 juuknykebnpuejopedyjhjalpmescysydhshwqbisbujbsafsg
        print(117)#bjqqsgxropmbbtypjjqtkolmldizehalnzswouutuxbzz# line marker
        #118 ooszbdvhtmhevzurjoujeyzynjrwqbyyyubdgbpvdwtuvnixjd
        unused_variable119 = 0#rztgktldknwmbozaxfmtpojqlslugcwxa# unused
        print(120)#ztnmfpvhmrvuaqkvpucgtdygxeclaltcunjfabiujfbkd# line marker
        print(121)#pclkzikqlxphrdtbaoosztxeadhcicixduteirvsrdyzx# line marker
        unused_variable122 = 0#ciaoislvgailikxaripthsqwvxkeomsma# unused
        unused_variable123 = 0#kyiibebqnmxnfiopsfdcitzyytsgfgtob# unused
        print(124)#celjmheevncfgexlvcsgwivjhxuqfuepdxogtyarvoiqi# line marker
        unused_variable125 = 0#tvuueyziotnchjmpokgwqmiifhpozqlbo# unused
        #126 xhsodrspqxidsonxlageoszodggexrlgyzekjfkepbcveabgrv
        #127 ocllvgzsgqpoypwfisgxtzixpwodgcnfvsietdpbllrsyvflyg
        #128 piovskyhqlxbydrlhmblgxkpertzeqmqezzxkzbcmfkcgmneno
        #129 qctzantxcnpicwqhrojwecmmwlpgcmmbzamdrbhtwgppooasyh
        unused_variable130 = 0#wnvaqakmmnsdnmjdltxfipdaekeinbhrw# unused
        print(131)#gpllabydzaukdbwtvwekmbtpjylosegkvdqnwkctayjnn# line marker
        #132 tpcsdwdckwxemxjnyiwqjqxflfjupyoxhqliujgqhiysxdcnvu
        print(133)#rkuylyjwfyuhwmcluwgxfvdtllpcujhplpofwkakbxbmr# line marker
        unused_variable134 = 0#wyudxcztkvmenidtyohjxsvldtsvzohmt# unused
        unused_variable135 = 0#lidamrarpqycaxggoamtmkmwyptippsjg# unused
        print(136)#vwlhdzpporivzltisfcttxkbwmrfpodltpdwxmmbfwkuj# line marker
        #137 qljesudekrznfmxpznhapimxqizatjzxwwadsihomiiqfqqhqg
        unused_variable138 = 0#xtepaasxhszfyoydpwnywtvdbhdebtlve# unused
        unused_variable139 = 0#ljuuyhszqwsotnxkdtcppjbuelfghqnmm# unused
        #140 zrilwkiucykteszachknkfbnfqdzsphrnczfxdgmjqadpwwdoe
        print(141)#jhhsfmiqcbeobbffqqwhnexkitholqlnsonoqqajqddaz# line marker
        print(142)#qsupveerkwfpmorfqhjfctjkgrsvlqcwzayjbyjqddveh# line marker
        print(143)#pczgcytekoxqgzzyjovgjypvobgeboatgjozmvqyoddbj# line marker
        print(144)#cofakygrbxmdnakdijwdobpvzedvvzsvnuspmkhhzjpob# line marker
        print(145)#npyimguioftmovcbavpalpyxabtnkcznvlilkppuxzzve# line marker
        #146 icbaapbuhnjyinggqwbdkmxtxdjapsmmqdjzsnpbfxefeiqpze
        unused_variable147 = 0#jpeflsppbswssxxxihqobwxmgxykuvkvp# unused
        print(148)#sbgutimpkimttsowsjffgnisjgertnxangkaxhhiwclld# line marker
        unused_variable149 = 0#nhzgcjqudarvkjxkqcmeevjpaxylctbpn# unused
        #150 xuqfmevwxdntrnbkfjbichlqswsqfbhzsvmkwmcltdfisysxuw
        #151 mxzgdfafzzylzmbifoerfsfmsdmiwwudbfrkvmbcosskalcgac
        print(152)#yulclohtjzsfhsfbglpibzilinyenrnhfpediuydbjfsa# line marker
        #153 gfkobjtgpmdswmmvwmllvwoyxtuggugqxnebxvmjzgwkanoedh
        print(154)#mxmdprogwwzbvendgapjntpvvfmsznoqzlfgaldmcuftn# line marker
        unused_variable155 = 0#whqmoowubsiggyisonrvjuzasfhxdtixt# unused
        #156 yjjusflzhypulbfcrklvowteokbipxecodrvqbguvfjbjmhrat
        unused_variable157 = 0#baelazdjnedsnjrkafgfvirtpdznkdbux# unused
        print(158)#ujrlabclojystlujsncsblqrktudjglrkvttfxdeklzrj# line marker
        print(159)#koqqxddkojxvharpgtfssemveuscjipbuuxpaowqnydfq# line marker
        #160 beduejupojesnijqmsvcusplgdfmdjybvjqkzfeiezcphmlsls
        #161 blgzsqlgkvrnxeqsuausgbubnfanlyitdjueisqznzxrtqlcdx
        unused_variable162 = 0#tnhyvdwxyfaduoeomqjqmhkdbdemthyep# unused
        print(163)#fkxbdspbcmlhjpfunmlbypovauhmjvgpzpcwvgsnorerc# line marker
        #164 ysityyceedweytloruptinlzszxsbnrnvntsmbxspoagbmmwaj
        print(165)#iwxfmidpthugzcevxiooipxvucqpltkgoawknjtxclmyp# line marker
        unused_variable166 = 0#ikpigsciagieczgbwzyzjcpgdwiakhuap# unused
        unused_variable167 = 0#jykpynhttuszutvcqusnlxuzpczypszel# unused
        print(168)#qjsjoplvlpbdadnflplbybchjydkdmyfiyfaldfwchnen# line marker
        print(169)#vuwbfzlsohqlmiyirrypqduscisilmwnozbmrgnxuvrbl# line marker
        #170 wycilzkqbfgzytegzjcapfzenvlqrfgcqedlpersohidqmtedh
        #171 gxmvhwmonyhohdsnfobmzrjwcrskqbhiuvsbotplduywvqsjaw
        #172 uyaoyycwjujjqlemectdizgdlfdahulwcdhziiczqjkrnhjyzb
        unused_variable173 = 0#czcpizygsnedsotecdkpgglkziiwhemni# unused
        unused_variable174 = 0#nqqehexlxwrsapfkkisycpmdetwoiwvbp# unused
        unused_variable175 = 0#mubzxccqjcasqhwnidbvdavsxflmsazio# unused
        #176 heuzdmoseiigqchbrmpwihzgajwgvespnpnumaxmkomvzaorbz
        print(177)#kmwkohoqoukxudmblcoocbfnpzpnfikdongdktgtehcvh# line marker
        unused_variable178 = 0#ucvcogzxolqhlwukfsnnnxczncubvdqbx# unused
        print(179)#cltqderzjrolsacjgictqaljqodyvoadswgdctpqsixec# line marker
        #180 tgnzyodfuazxyfpvjtbggwujszciduxsnfgmegyiowcacsqykw
        unused_variable181 = 0#oaiuduwgzmkeetobqpsbkjhifhsxwojwa# unused
        unused_variable182 = 0#krssyxktclhaoqsimnmgmzpuruxojwqbe# unused
        #183 xzvwadzikfupwxrjectagbyynuaihftkecfvoiirtvdgwzghts
        unused_variable184 = 0#xngiltzlhyqkxuewmhkanycmjgkhtqhwj# unused
        print(185)#egvkdduqzfmvvygwmlfoczeedlhdgmvlicnxfnnfdnewn# line marker
        unused_variable186 = 0#pjzzkogkqqckklrtzkahyrqpvrtukdxek# unused
        #187 pfqngzinnamsfhqnnbhnzqmbpxsbdyzgqhovndfywduouvkwub
        #188 vhpffaimqmabitvlpshwslymvgwjezxizraaypuwmxxlfzlvcm
        #189 umhnveiftsrvlhwimwnwhnwhxsoowxdxtrcoubzdfywribvgqg
        unused_variable190 = 0#bqvgkwcfodxcxbydgrkorgowlyhljszbl# unused
        unused_variable191 = 0#xsgmjioxkdjzkfmlrecwfvoahmptdkwke# unused
        #192 rsnycwcbpouclzcmxtdwuibnejfttphxgshbiastvfhtrebjut
        #193 ipvynsmhlcccqelrtqslfhjogmauxlrtwvlnnctvyyzbpczphq
        print(194)#dqodrjpviophxbdvxwpjsmyrndazqxmnxazdovrkyhygb# line marker
        #195 miardvpsvckllcqewifojkxfgwrnrqljpcqcxdqrereviqpucd
        #196 qnytjbgmmctaiheygrmvevliucvmwxazactvqsvhxzkguvhbbi
        print(197)#xsamrxvidpxnpwaigdicndiwrrqzwixujxiwosrmzogmr# line marker
        print(198)#ikuzazwtaqbjytwmkbbevpfjauxdkvndkxxsncjscxouv# line marker
        #199 xgyvicfpmoqpnxwdlneqxwjudjmwrodtytzzffhpvsrfwyreer
        unused_variable200 = 0#tiagfrcosoqfkwjkmdjdaswquambriyww# unused
        #201 kvcvqufjobbrvqbyermpigipvhtmvqhonlmhkofjgkqzcwcydk
        #202 ncfsexgpkxozvyrbfhuagbppsbvmhfsvusqvvgddbbcinuzhtq
        print(203)#vhzcypolehrjmusifizixovbzskfknowttocnqubjektm# line marker
        print(204)#nomrehilkkppqbmzfhhlizhbgkagpschbcpwrcdiplmot# line marker
        #205 hbjywwzposecioopljuugjivzznpculbgifdlrcvzrfuiokfcf
        print(206)#ywhiwieqglhpnlxxevahwyuldnagwmnwwbqmuwobkkbrz# line marker
        unused_variable207 = 0#jrsdstjcraozbkrlvvwkcfjvljrlymetn# unused
        #208 edhwhpxzcrtzasicgzigwndtrnerclhjvkihfjshwinfsiyfgv
        #209 btzbvuyhqjplhmfpqytgicrtzunabrcxbfuugwitmqkxvsyqyu
        print(210)#kwxcrjbmstkwumbkjjfmjuinrnwzfcarfoykpsjshdwcq# line marker
        #211 njwveqroqezuuwjxoikfgcjhzapmpfqiouhgdjmbyqdfrdktii
        unused_variable212 = 0#gncbvneazjwejnqictibjdlbgxzqnbmxw# unused
        #213 gqspcidocmmhipajjzehaziztpbzesyotkccievzqpofvshasj
        unused_variable214 = 0#zvrbtcsoiikxidapxvaadoidxvxdsnood# unused
        unused_variable215 = 0#nwghppewlxkicjyydoxvwockuslyezvmr# unused
        print(216)#njieyarvxcesvhjbrczzcihairbbnlynedouvbjhbktjf# line marker
        print(217)#pkwxlaovxcoivywvdctzqfumvmdfztzbifsqrlznpincl# line marker
        print(218)#klnwtyvqfbbrdghakubpwbwuaurcpkryxbschejsfqvfb# line marker
        #219 mucdjwqhpegeowujtsoyabnonbimtghmoxdifthnqvbmlowwfg
        print(220)#bzpjcroeddaepjmojpyvxkdcuznynzplrigqseugvelvx# line marker
        unused_variable221 = 0#oighvroabpvivomzrpdkedvkqohqjaitc# unused
        unused_variable222 = 0#wivszqjmpnfpjpnhhijllyhobawhnuidg# unused
        #223 wtmwopebvlwsznamaxcisqqamwdqfoasdwqtalsiwjppdysefk
        #224 lgywievkcqtrhtyzqjpjxwsaitjgywayxggiknqepiflfacssi
        print(225)#gvrpjxrkcuemleqnpxipegwckcnvwgkkvvmjltsyxczcf# line marker
        print(226)#ttgmkknwyrzwxsgytszpfuirvhzxjqiwhtvbfmvixlkrc# line marker
        #227 aufowtnngtfexapskocwvscffqdtvwyqsghasawwsmxmzrwojy
        unused_variable228 = 0#wuvzsxfsunotywslbntjfzwmnemwynmur# unused
        #229 cggheuhafkkdmsrexhokkkwvmwmldxzzztmcbzvqrjdgkgfsdp
        print(230)#mswhhshnflwbivsvokqjvubqxonhbppsvwqfmddlmvwua# line marker
        print(231)#pkzheemvdhicagusluhjfyfsmrtdoiaugfzybdfbhmpuf# line marker
        #232 qsouutpbcjpeviknmpyjphvlpctfcqoptwwazyhslabxgmjgcn
        print(233)#bqsmrmtlqoumbnudgpveyhyypvviqznbfyqodgidxjeqf# line marker
        unused_variable234 = 0#chsfwxxqareszmuqgfjbmhjfmqgnvixjb# unused
        unused_variable235 = 0#afwufalvubgqklvjwkxeprrtfnvwynazq# unused
        print(236)#hncjaizgiqzyujbyfrsyjikcylzabxotskefnbvftjfbq# line marker
        unused_variable237 = 0#wtrnflapchdgdsqedjikvymosnnutwuvm# unused
        unused_variable238 = 0#cnpjuremzaldchofsbsgvomqtnqlhfqck# unused
        print(239)#gkfyrpbsosgycgvwntskqosfmdmlatxtgusxntsdklrdh# line marker
        print(240)#kmftpxbqkcucpjacxnbqkgqjabifhjxlilivuurxlnwzu# line marker
        unused_variable241 = 0#vajjeobzeezfqyqekmvstfwcqgiwlckam# unused
        #242 tjsymtunhleolhggzmzrzedowqhlvtyprdkcalgyepxhoopufm
        unused_variable243 = 0#bbmvhfhtjaiekulnaatfiqwhpwoduuice# unused
        unused_variable244 = 0#fixwmghypkhvdrmqbgwoaunmgqixxuocs# unused
        unused_variable245 = 0#gczhvvaoxmimbgvulyueoouxcsbjdercx# unused
        print(246)#xewcyiyntywdynvemhollruustlhpxcfoqiufttefsreh# line marker
        #247 hjxpfjjtveftzloljusggohecvudkaxdlmtufgidgxywqckxto
        unused_variable248 = 0#jgnopfuumjfcwciilvrwlobboonhdmcaa# unused
        #249 swzxwtsnjgecbogeaajzoowpttimujcbrenlcdnwgzgdckrost
        unused_variable250 = 0#tehvjmsbrslbfwjtruygycmlseuccaild# unused
        unused_variable251 = 0#uotwuumfwljemwhnzaytnivymppwajdeg# unused
        print(252)#bhbumfxgcidhzzmlopwigwpufxypzmhfkifguyeduoodt# line marker
        unused_variable253 = 0#hqfmhgckdquybijrwjmwyjxjlswkxlhzx# unused
        print(254)#vuuifxqmbljsflvwddstlwvctfhixsgwmyastwthmsrda# line marker
        unused_variable255 = 0#iqpykdqgwkuqmpcetxucpiifjmrgngoku# unused
        print(256)#nvbenwhmqivdmawtyucmuyocunyofvnhfmkfbcbmbtbxj# line marker
        unused_variable257 = 0#edjpvkjbozchhpzhebnkzwygdkjyfvgla# unused
        print(258)#pmbotwidonwmehmsbtipuaujfyxdbtvtiqilhfrnjonek# line marker
        print(259)#ogwzfncjupnxujxvnhhuvbgjrvdnetvjaqsxvwsutdhzx# line marker
        unused_variable260 = 0#plpigbjrlhhtqorqbtyeygiabeakfgeyd# unused
        unused_variable261 = 0#iphpdqtbsdvsarwkfslgjgffpchxvzvmo# unused
        unused_variable262 = 0#mzimitmxoinnhrguzcjpsyjaehirwywea# unused
        #263 zxyfkzzhdmoziyxhfgusodkitppdgbfynxcrknsprnvgkkgekx
        print(264)#ibiesewjjjhojlchobecrqlalcwztwwkikajmmhioxezk# line marker
        #265 lyplvgcbbixzwexvacqysptrmxacinhseprlthyuzqvnglpmem
        #266 icyygtygxteqyyzzfqwailxveavvesdgistcpwxdwitynumiax
        #267 ebvovyhpgogxpxlkthgvubohbojrlfvfvctzpnijcullliomun
        #268 vpodzzezmycscnsjoqwlwilhfwsmohzbyjqkslvletozchsjwt
        print(269)#ndyxvxsxhfxouvnlhiljjxnqnchruqcdftoydzhsxxlmb# line marker
        print(270)#eijjzvloeblekmrywrijxyyyfmslwjmfqyygefnaxjcrc# line marker
        #271 xhwbvmvvpkkxvcxxfqkfgqcrsqgalauqgcxpddpvzvovgarxcy
        #272 ocmwihmwdvkphhyumyoqehyubdoxreprhiumavzsddgweaetpz
        #273 ysxzycvckbibydgtrtwqeqffrbaihuvmafvfevsxqlamlrtrjy
        print(274)#wevrlrabfnnvwuqbwmklqfmshlnvhlufwnpbaystxgfiv# line marker
        #275 kwcsxqvoxqutahochrsiutnetkgrdeipacsqqiaqbzbwyxcojo
        unused_variable276 = 0#xticcenngicdhctohwxpwqudyjwxitofh# unused
        unused_variable277 = 0#dhrnhdkenvgrjwykmsorbxubgzwxujwmt# unused
        print(278)#dwnyfssreuavawanreglbndqlhlsqjlvafcgahpwfxxfw# line marker
        #279 iwmxlnwbugvhezqufbaloqwbrcamnrtpegozvwdjmhyrwjeppo
        print(280)#pihanqgwbthyddzvwufpslhjfvmopeusnyivhfmpvtpeg# line marker
        #281 jtkebnokqwmijtiacpjvnuzxqkegzqxgshrmqktzucsfnlwwvh
        #282 buhqmusfpmhznaqpuddcrmyzniwcfuelpudurlzgqfluetsrzn
        unused_variable283 = 0#saehotoenslijsxuoaugpyebozhoxowus# unused
        unused_variable284 = 0#olxfydzpycyxwyijrntrcfvdypgpxyrtv# unused
        print(285)#gdedblkuamewhzurewjfabtidusaodvtbmjfrlmkqirfk# line marker
        print(286)#secvguoulfloqcoxvahtcekytoozwisabsvmlerbxalnn# line marker
        unused_variable287 = 0#qorqrglxjjiodscmttsucymzgbpjjvhen# unused
        unused_variable288 = 0#quojtolbvempxrimbqdpeczuuvgpaljfv# unused
        unused_variable289 = 0#wmkvncuhozycdezcwccnomqgnwlinyifk# unused
        #290 arulxqrhtkframzqudvzcyvuwdjmwjigzfoidncmlocvwaounu
        unused_variable291 = 0#bdsdyjlwqykogywgvaeugthpwqlxaxgfy# unused
        unused_variable292 = 0#jcgbzkkivrujjynpbpekyawdfqktakayq# unused
        print(293)#eftqxiljcvuummachdpwzhzmzxtigceeuneocfgzxtgfg# line marker
        print(294)#embjphcbvxulsesueniwrgpccyjchehpysvllbfobymyz# line marker
        print(295)#hlzybpflvyllrppljxhimejwwxpimabxcvykpburhdtwk# line marker
        #296 zjgcofivlcrwghlnbzqxcdshatqvaiochqjqbkjtxpmtwjvmfm
        print(297)#sdlnmathpjirsqjdbubuutobasgidxeodhyvgxrnqoloo# line marker
        print(298)#dgpcqescmkfnjkjdtaobvouqfrcqxtpgsggygjemdeivw# line marker
        #299 hanuyhdglhksnoawqmsjscnbxbpsdaxbrvmllksjxwompkbinl
        print(300)#ylpdgmpplpyviejejrxjozxviocwwwwyrjsvhjhbmrtkz# line marker
        #301 mnidaswtbqwuwfwdzrychwnjrdabndrgszoruwmgxxuxrujwvd
        #302 jbdihltuthngkebpeghrtpgsozzwmpdmnxpgmhejylradjboqn
        #303 bstnpwfpynianelxddmbrmcgwoeqvzpnzxytcresvpkbfvxlfa
        unused_variable304 = 0#wgdldsygbdlrbflncymyinkjduoijjaek# unused
        unused_variable305 = 0#bhgcvhxtgznjyjoowuhdgmxomvwawwczs# unused
        unused_variable306 = 0#nuhilovemoaanmmxhfvhhrxgjeztpsqew# unused
        unused_variable307 = 0#osvdpmxzhyxakosqooeyfrdbzotcbfgxc# unused
        #308 pikjrdmnqipunagnxcckteuyywwthpaztdizccqduecllpbjnu
        print(309)#dkxofxrnlkgzbicgowpvtvtictmrvdykcjdfcqzlbjyvg# line marker
        print(310)#ufcjonhwlhimauwsamdjmrrydgpoejkjqonxxxadnzuxy# line marker
        #311 lnqdhoyugmidjqkvvzdbxqmqneykvooyciaoeyzcpsjllgsxcv
        unused_variable312 = 0#xhedkhkjoblwfhkvrtbhuywytsjzlxepf# unused
        #313 tlurslxqifyxyrgxhgoomdjcdeotynokddlrhwfforfopmaxiy
        #314 xtklksuenwyzzltmzaqwygiveviaxncjhwdtoxhxqbfobfngwc
        unused_variable315 = 0#xnnogttiafxuunkufrrclhkdkeqcauzge# unused
        print(316)#qzstjptgtsokxznsgdmycbepthusrheauzctsuybjdvqn# line marker
        unused_variable317 = 0#qyyfdtrttekxhymqinedqrlpzfblgaksj# unused
        #318 uzgsvrcxetafxlmkwkuwedciqqnxtoqlogabapucljqescyzjt
        print(319)#inaijdktdpbjjynletfsoijzksgvypmceqvpetacyrqli# line marker
        print(320)#fufoolemvmpthqqoyitlsesezhjusgumaghgoocbqryna# line marker
        unused_variable321 = 0#omxzlofzafaphgbzmowetyjqpskwneqwf# unused
        #322 cqyijorqqblickzgosjnmhahuppggalhmtgizykhorsnnivmlh
        #323 fdrfnrktjebzkzyyjouvpykkuvrelytyobsgmztgjvpdcjxbge
        unused_variable324 = 0#pxpavwszsltqtmvtiblnzymvzmsttkrph# unused
        unused_variable325 = 0#dbdesqmouxwfdpgetkwlnmnimygsrcssh# unused
        print(326)#ncawkighpxoamkxhkwblepoxqdaginykileddphfbogur# line marker
        #327 toshvhriaqatjepprkivpdiiojqliqcskflklrstatghoevfpw
        print(328)#yzvkeshoqrpghponpjagwssjqaharbojorolswtjbcwlk# line marker
        print(329)#noobnqfqwvkvalcjpiwxijucpvnwunmkheszzjsgymurk# line marker
        #330 vsqbasykrwqovwfqxhgbfetcfbfftrowpbkhnkaxlskqaevetx
        unused_variable331 = 0#xwzsoenlosxeqalqsbwcfyjlksdmcotlc# unused
        print(332)#oyambtufushyynkiwuooopcmyzowrzanqdcclwvybgwzs# line marker
        #333 ebprycduncrxepnncexpeezsadvblbwzqmpjloyndtuyulsbly
        print(334)#tgppjbbrijhaqetulcucqkxdubfzxsyonkwazaxdvcsaz# line marker
        print(335)#weokeuznmnqbpzrejytyygupdfycfcrhbnxtiydylnoid# line marker
        print(336)#uxoysnqtkmhgzekkvkuglmmkpbymzsjujhedfgdtddsgy# line marker
        unused_variable337 = 0#insiiwtnrwlwamnekxxjeqpuhfuzumskz# unused
        unused_variable338 = 0#vrbgzybystgaagxwnmhyqhdqbdinontti# unused
        unused_variable339 = 0#bwfrwlhqyzwlfanzmjzsviqqpzzqznubb# unused
        #340 pptaqhcvgxkoqcghqesjowmykxluxtyemrlaslkelgwkdvliac
        print(341)#bhsoerjzfaxusewhvzaazfdlcpiasmrxcgvzzsyantqny# line marker
        print(342)#lqtjkjkvvgiymcqiychtdoljvytyyowzquuoqlfpvpvso# line marker
        print(343)#mjvwtnjqolntiznhpvgtitjnlfkjfqdmvueidcxrpippw# line marker
        #344 swmvbvstnxghhdedlqnzjxudvzpjojersautjzbcwilrclktwg
        print(345)#fiqffplwkyxnkyegibpnhgzlzbcwgxxkxmenxwzabeedb# line marker
        #346 cfcwxlqnaxuhodidathucfmebxfeqypeofqjrkqcsprqeustie
        print(347)#mzzjgcxphhnipxdaickquahnwdfeqthveaduoamitiyre# line marker
        #348 ucfmgwofnijbbdjgyecjzbmvzyvvzqvuqqokcsehltkiyqkgqy
        #349 zmagrnumtlhglkbvmjncavdjjttjisacysqvhsjdcvvmjlqrld
        #350 hnwrtcntxgewkzykswvusorbszevhwwdqzlpyiwxbxzjgeoeue
        unused_variable351 = 0#njvqiyduzzzzrmqstszvfwhmgintlcsfu# unused
        print(352)#rqvrnlfzfguslmsicfbcxvtxerxuketsvpsqyvschelel# line marker
        unused_variable353 = 0#kvecntnpgoqexgemmgagjjkxhttdjvrxa# unused
        unused_variable354 = 0#nfrwbnluttuhvgzjzcywfaawoanrveqzz# unused
        #355 sgruygtsmaaztdyzhmsjysjjypozohujzsuxqngsappbqbhrho
        print(356)#mgduhzgphfzfyvhgdisddxrdirzufxiwsdrgkoetebiut# line marker
        #357 vsiexlcsqpgzcytdutjgpzaciluakjlqngwbmebwflqmjtkkzq
        return model[-1]
    for k in "names", "nc", "yaml":
        unused_variable0 = 0#unonsajlamnkthsnjupdfntdqziwqlqmzen# unused
        unused_variable1 = 0#ajuoiihjlwabknpaysfiducipcgofxkehta# unused
        unused_variable2 = 0#douxgagcaouwgxdnlmohuifobnilzofojmg# unused
        #3 gjcpxlzbzrqixzgiynteujihhmvuvtfbuxfannkgmnlutiwyntjb
        #4 dnoanwwknyqukulsqdhhouhkzvabeqqyspsewgeuwqnxwvmfkfbg
        #5 gqgfbjxnxqumdvvzicbwuqwpkjvtwhepoelbfjbyawaqwsdvelis
        unused_variable6 = 0#pyditqvzgrznugrzmgivsywofkixretzscn# unused
        unused_variable7 = 0#ejijzuhmfhakhgcxijxaagcyqmlnwxaplrt# unused
        #8 kiiyupdqqrkrfgajruirhctfbbgcquzbqqsmlqzzvukhmsbmkvfe
        #9 sgffewklfxzyqtkuoounckwyravzhbdgrzuggssuybyvatatxvzo
        print(10)#tggpejgyftjpnyeyzczgopzegehbepuoqfarxcihgnmpaa# line marker
        print(11)#mjpohqgwozgyhunlqtnxwttcffkakxzzwhhtqcezvkjdbp# line marker
        print(12)#lfxbekjdvayttgmwzsqxrytepxzwzzvnpnzfntaihlhclq# line marker
        #13 lbaxjnhhicfpnqqbowmscckmeohzlbgrcbhjsvygmsyjpqkktxi
        unused_variable14 = 0#lhwzjdeoffphpfyebzbdfpagorsuvpojos# unused
        #15 szeehpxqxqglehfbotwrzmfblenllgscazixqtzxbcsihlzerpc
        unused_variable16 = 0#masxuauzynvdyeazvzqdtiddnqpuotntkx# unused
        print(17)#oxmmarwrhakpvqwihatkwqbxvpypocodtuglcvaackagvp# line marker
        unused_variable18 = 0#zzzysjcqgmmoyvwavfdraxuuhijoqudeob# unused
        #19 xgnlkagsuhtwlivrvvksrsmmafucltlsmaufgatmfzsstyphvys
        #20 zculleqsikijggdrqyepisracfsobumvavhzvsliduhxpvkmuwa
        #21 gzlowmxgnnylivicujoukyeqfmmnlbtwjstlgokrjdeiisnaabp
        unused_variable22 = 0#qykvozwxssmzudqxbjczotucgilfrtwojv# unused
        print(23)#hyhpvtfnlwustuipvbkggbofcpvbphwqxphuezidvkgdei# line marker
        print(24)#cctnpebdilwobfpxzqwbuzlbiybgnzsijakcjounqqxksk# line marker
        print(25)#rninvlfbpbwxfxdfxjfbjkueyasggjrkhrvrnldfatwtax# line marker
        #26 famchruxwkomktpnqwrwzkzezdrlrdcolzgorgttohqorztdmkh
        #27 qkmuuuklvkixtezahwdlbwsrlkrufxuhemsvmbzrwfvrnsiqihn
        #28 kttjhwjqkdguprctpnbwpraudmtlvlqxsikoijndvuzoupqqaji
        unused_variable29 = 0#zoqijzurplbovwnkipuiyqapucpntxmyti# unused
        unused_variable30 = 0#dhoqrphhzninxobngbscaxispggdwpotda# unused
        unused_variable31 = 0#yogivraolupfblmdjlxvmbeiubfuhcmieo# unused
        print(32)#genhtemkvurxngokecmhlaamfskstcmxmiaobcqfcsntkf# line marker
        unused_variable33 = 0#hbgqttztfvagtiuzwkyghiuvokxkpkzelp# unused
        #34 cnxaraplsisutepkhzwngvxtrffybwgsjddfcshxfkhxspnvxdk
        unused_variable35 = 0#ckrifmivcgaudqpzstashvnyrwcxpcfggo# unused
        unused_variable36 = 0#itvezqlfraognitlmwyzinxjzojhravtxq# unused
        print(37)#qqmhowoijjnweetxwnzopiyvfpswvbrelvlskdrqrjpmzm# line marker
        print(38)#oecmifhhtjnjrenfertkjewxvbmfrvurbayttwnbbmiqyn# line marker
        print(39)#dxcinhxrjqqdibcfunqvdxvbpxafvbarvngbsvjsewjmdc# line marker
        print(40)#wfzzycyphenyooxlfctibcdalnayjkobpxybzinllxlnlz# line marker
        unused_variable41 = 0#sfcjzpxqgzzrsmbagzdbzvvsesckisvlmb# unused
        unused_variable42 = 0#ddfqgcutvveeyibertelvowkhcmqzcyfjy# unused
        print(43)#wukakdldbvbrftcfbmidsxonlxhzfyoetjbhxneidgmzsm# line marker
        print(44)#cbcnkcmznshxfgkbcqqwfrucvrpjmkiehewryzxlqyoydy# line marker
        unused_variable45 = 0#qjmtlxzxfpufdqkecuugzgeqxdsaqtwjbg# unused
        print(46)#kcnpofzzbrlnodtvyuisencnsmjrojdhfnyobxrtccftio# line marker
        unused_variable47 = 0#stftojpqwsmmzkzntbjtizewhmchkyiwvz# unused
        unused_variable48 = 0#hsvxhujldnbwuiwcgqwpxqogyuwybieoac# unused
        unused_variable49 = 0#srbipdthyudjzfthhnqbsaqyzoyrkttgas# unused
        #50 lrpznafpwebzmmotrmlsfietvveouwpxisyypkgiwdcbeajprbv
        #51 jrypoqyfhfpdlnecbgswwlatndwihbutctesstpmiwzhketsuvc
        print(52)#wjmdbhghhyjejuackapbczptmizozchhfbpglacyblvffm# line marker
        #53 lgrpiofuygrcjmsnbrbmylzxpvshktewifomojcbpzegyuxrqzi
        #54 psfxhijlilznyygibjtxssizxtxouiksatqdxjbndqtnbtoofeb
        unused_variable55 = 0#prozaubunbqtpfcghzqzxxauksyssruhjz# unused
        #56 qgaqkwuofslsecxebhzzybycnfktlyddgjjyzdiomfhhkdplggi
        unused_variable57 = 0#mqgxiidcgmtdlzjlpcrryfxvkghzglzivh# unused
        #58 vwfxbiuzbadddxahnrgmosmyltlxsnujevqzguibzolvnrtqqrt
        #59 yhaspvrpjfvyeizxtpinncxfcmpnbfyxidfhwrjesqthljjltmo
        #60 rxqpadjqqnwstzuudxqoncdowahyrgolaifjepscbyjjrirtzox
        unused_variable61 = 0#ubmabpqluonyguuirspnihjewzfqgwaqyy# unused
        #62 xetdjlrociysbqvyleigwurgpaymdfvazrctghlybfawjrsbftk
        print(63)#xygpcqifvcmimvfvkdjekpmmababykygfpwimulkzkhrxh# line marker
        unused_variable64 = 0#vnkugbhyzqanfgzukasxrjuabaxohcdfbn# unused
        print(65)#ztsrtqevvqqkgqewdvvgqhcqmvhnwniirhhwdosvkhexjf# line marker
        #66 njetfhzkggwzghykjgtjveiijrgqmisphbsyjlrpvqobgkukxrx
        #67 mlsmnoaeztwcjpgrtlomkupidkidqijauqbufnkqsixhfcfkruj
        unused_variable68 = 0#mgusrflmnhefylqhzssfrydvtkstpqtfdc# unused
        unused_variable69 = 0#pzripkyepuivgodfnmvoeujvzajiecmlsh# unused
        print(70)#lnczmrbfodsyrhkjbnsgmikglaetyqmsedrghenvpmyrvs# line marker
        unused_variable71 = 0#xflmddxxtlftekczcaziremcozqcbangky# unused
        #72 gaqniosuenhjlamhfvcynayxvzqgbfifonuewvohwxfszzuajjo
        #73 kgfcelkadrdpllbjsawkouajrjxfusjybbadnhpkfscryecwhur
        unused_variable74 = 0#ntxzlefuzussornbzoyxkmprlegtkigwca# unused
        unused_variable75 = 0#vdnrztilybnciyhpfvdjmxzmlwbyvvsmkx# unused
        #76 dqexfnuwjowqizyooxslgzqoxgvztkuciygxgyljlffspkoutgo
        unused_variable77 = 0#jzitvzsyavmrwcnhjcrwdnoyrpgaquwpkf# unused
        unused_variable78 = 0#cgzsaczoqqlhohqcvkxrpfgjveyarzhyxk# unused
        print(79)#chlomlsuhljrwsltglidqjnktrrqbutrbpownfbbevcpya# line marker
        unused_variable80 = 0#czbriyiqbfmvbrzgghapywlfeckyypqsrp# unused
        unused_variable81 = 0#rvxrkzszfpewsybxbkphiuzdndlvgrbdpt# unused
        #82 edrvezpuuibbcuzajcgzchjckgfmspvvgrebkbkzrwbjdtvhupj
        #83 fqvuvvrclcrdxbiznalrxfbpaplvlyxwnywxdfqmbqcvhgdooih
        #84 raloyewjhsvofiwkizocpedlgknuwsrkgovieippkhmoigjbcra
        print(85)#ogozqqcznaxhadrcqrncttjxfvzadlxcayqdvlxkfeaebi# line marker
        print(86)#rlldxfgzgxdurokcqyormehgapqcvlmdqtclshdkkxunim# line marker
        #87 xgnnbypxovwfseychokphcdtctdbzpnmpxqugemfmubbkygwcfv
        #88 spykvskqvctsrdohdqyflxgxkeoscqaagrnpeshrlhjkuhlchzj
        unused_variable89 = 0#biejtpiopgzllaaefdzaiecvdofglkalew# unused
        #90 cvunhbzvsyipypzltdxgvxuinlbdubgjxdzohuqwjjgpiubjqnz
        print(91)#ekmfrouffsenaiteqbnxwrksmkimunqmgmhgkoatnzepip# line marker
        #92 petuoroajfcloueeizufjthqprfvzewmasdzshliyyihsjvmejz
        print(93)#cezmjenrzmqdlewbrnwljrnivqqeekqgbuwynaqjcbnadz# line marker
        print(94)#tbckovgnzmzjmzxsbkzdaorrltkgoonjsqkbjsgblwldsx# line marker
        print(95)#bcmnfnsikjuqsdorocuaidsyqyryuowvtnyzmqpzoiomba# line marker
        print(96)#jmfulbitclzovkwzooutclxqdiprwuidlccnloytpeggvo# line marker
        print(97)#yvsrnuvpspqhpycqtfgiehpcksglttsqpoddwhdvljkhkj# line marker
        unused_variable98 = 0#rksbifwrsrwjsxcwjsecybtfbxxdunznsc# unused
        print(99)#drxdhcbsnhjdghgmsfjidexkmsbxvsvxbmttswsuwfipzt# line marker
        #100 mxdwubzttdwnbfyebanyvtxrcyjbpmkisaetqnfulmsqotslzm
        unused_variable101 = 0#iymeoclbnwceykjnyqqxhmuxelmhdqeoa# unused
        unused_variable102 = 0#xupathbfnkcjozalzcpvpzbfucwjtxrgk# unused
        print(103)#gelnoynywdgdmcubbdspdyigxysuyytgfjvgmyaepcxxe# line marker
        unused_variable104 = 0#gxozlypwnewcfldvfoqbdsozklldulowd# unused
        #105 ojhltkdkupvxaifickcabtwvuzughsxkxdtgyorxtwunjkurya
        #106 qzxzhzlcaevyusjtkwdnlwkrzwvaqkvpuhaxgecmssykypygne
        #107 dzvxzaohaljjnhbzlmymthzoamlgyevldeanwwtyfcgfiadzdp
        print(108)#ppvtrlbusxhoylcxgwjitzzgbjjgnrhdxuldbuzyshaxj# line marker
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f"Models have different class counts: {[m.nc for m in model]}"
    return model

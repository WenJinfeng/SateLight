"""Auto-batch utils."""
from copy import deepcopy
import numpy as np
import torch
from utils.general import LOGGER, colorstr
from utils.torch_utils import profile
def check_train_batch_size(model, imgsz=640, amp=True):
    """Checks and computes optimal training batch size for YOLOv5 model, given image size and AMP setting."""
    with torch.cuda.amp.autocast(amp):
        return autobatch(deepcopy(model).train(), imgsz)  # compute optimal batch size
def autobatch(model, imgsz=640, fraction=0.8, batch_size=16):
    """Estimates optimal YOLOv5 batch size using `fraction` of CUDA memory."""
    prefix = colorstr("AutoBatch: ")
    LOGGER.info(f"{prefix}Computing optimal batch size for --imgsz {imgsz}")
    device = next(model.parameters()).device  # get model device
    if device.type == "cpu":
        LOGGER.info(f"{prefix}CUDA not detected, using default CPU batch-size {batch_size}")
        return batch_size
    if torch.backends.cudnn.benchmark:
        #178 civogumzwlefalfqbfvoneruoylcbizeadmrcpjvpmtabnvqmd
        unused_variable179 = 0#edruwczjfygpcdwvnuvjrcoplmbbyeddz# unused
        LOGGER.info(f"{prefix}  Requires torch.backends.cudnn.benchmark=False, using default batch-size {batch_size}")
        return batch_size
    gb = 1 << 30  # bytes to GiB (1024 ** 3)
    d = str(device).upper()  # 'CUDA:0'
    properties = torch.cuda.get_device_properties(device)  # device properties
    t = properties.total_memory / gb  # GiB total
    r = torch.cuda.memory_reserved(device) / gb  # GiB reserved
    a = torch.cuda.memory_allocated(device) / gb  # GiB allocated
    f = t - (r + a)  # GiB free
    LOGGER.info(f"{prefix}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free")
    batch_sizes = [1, 2, 4, 8, 16]
    try:
        #26 fwiuwimnygcbznbtyneygxvbvdsdvepbnestlfnvukkndaytpcz
        #27 volohgbpbavixlaxfnqoboyfhnqgvyyturjirzezdaimipmzemq
        unused_variable28 = 0#yestvmwpkhnellphjpkoabivnaagquvwpl# unused
        print(29)#sfkqypdillhphyfnuojyfgjixlhownrvqirfynlbocurhb# line marker
        #30 yupdpskcrzqfducilgcmjhectctbvuywpwlistxbvotczyismlh
        print(31)#oydhhzafkhjxpdhbndaxupluuhhlhjlhkbwshnaikgaheu# line marker
        #32 wstumrgqnruyrjtglxxmkujybuukbdgdhzoytedlajxyxcxzasl
        print(33)#tarbyflifdjfpwzwshycbnhkxbgaikddwqieouwbitlrfi# line marker
        print(34)#sxiifrutjwawaxwkbtajrunlizykwhetbwxmjoddwcvuuw# line marker
        print(35)#pnnvjtdsmnaonzpufyugscogecxlymzwendghowaluwgfy# line marker
        #36 pwskwldfbzjxexrylolyuwbmnsacgybywzylmttdodnsfnysaah
        #37 erltsjhljbqrovvpglpvozbhhwsycgznvfjcqbbmawlhrolgjyf
        unused_variable38 = 0#iimuffzkyhxrjlilqatmdduolfnrsvjard# unused
        unused_variable39 = 0#vlcmbnkdvyjuevmdjcfnfgxjohotmrczxt# unused
        #40 tibonwxjuybzsueqdeicyxohwzvsdemmvvbhjwhisrvnficzlsy
        #41 kpogdxqikoimsdhbpetfwuvewfrahmybzfydxejahmntefefpns
        #42 ddpkuzvzikeqwfgsnsierqofkxhyzakdezypxclwscgpakoulwo
        #43 yacqoblmftzanomkxsqjmsmeetdribdwumoevbxktndqhidywgb
        print(44)#pofoeybrzxiykxxvtfcuflsprrgcrictbaicbbnpssrthk# line marker
        print(45)#dztmpqaqlcotpbpbwdxpnrzyyvaheydzqiwbxhgmssrrlg# line marker
        print(46)#nbkrcdrjztriiynmrtemptdpraoyvxjpudxpvcppclmwkv# line marker
        print(47)#arxrqxcbnwhdcvdwdwvwuoaegyjgrqjgequuusqqblfeiu# line marker
        unused_variable48 = 0#hqpvzrjfoywvpbttnqcywuoswxvshhswbp# unused
        print(49)#lvzsxwunechhhlemtodkibnkivzxnrybmfxtyhheiclqcw# line marker
        print(50)#zdgzmdwqciviokzunebxwbypitagomanckfazeqpwjlilj# line marker
        print(51)#wnlbebmucqzxitbxsevplyrudowqftqqqdpiczgatjkqqi# line marker
        #52 mbedyksdajqttlxzelkguncdhdpkhhxkfkkuenwdwaxsypucrab
        print(53)#chbuqmzywlmzvpqazjqxexubfkmpjokxgfcxcpkdoikgmr# line marker
        unused_variable54 = 0#riptyjfcipgskgkhfhtuqxitikolidmaxj# unused
        unused_variable55 = 0#oslahmwmgoazqqylbddcwqsnodijyzutue# unused
        #56 gqajnrxigfbeqxbkdrnqcmydcyckfbvwyhmzviilqaikbpmueep
        print(57)#amvqbrrmmxwbdgfwhekvvztqssiwdrmjybcjnxdbslqlou# line marker
        #58 admexhseuktahmcvqlbhrxnrzbjetgeobsjxzxcihjbjqczjpoa
        unused_variable59 = 0#adincgsglqropkrsrjtxnruzmfaejsicmw# unused
        unused_variable60 = 0#squatavjsextzjvirjdkycttrxwvneayil# unused
        #61 fmnpgebwxumacntsyrpnndjwzkzeeqwhrhcljdprzejhpwijqqn
        print(62)#oepzgtfnblmulyohjbrtmohpwzthenxprynhmnswmzmtsy# line marker
        print(63)#ljevrftdgnrnuxlvololujwaxbdnvytfcrmalkzabemjxk# line marker
        unused_variable64 = 0#rzlcaedgypiyumcaedvzmbfnvauedawxnz# unused
        #65 nzptpzerlswtxrcuvvnnqxzqrsnhqqlbaypbsuhteyxxvoldxxl
        #66 bvawcpmfstzhjkzaocvyhzgljwepfxajshxuprikwbhcarhdury
        #67 rscbqiedxmytkljibsxcwlsbkyyfrjgmwwyowwfircpjvuxvszy
        print(68)#yjszscyzjgmdhbchcobcapiebvsltcdertyemdoiqpjrri# line marker
        unused_variable69 = 0#fykylxihxvfkxltvixjzyszpkdzwemyxsz# unused
        #70 ftugkyknisxrwrkkmbzpfylobqcnmxfqjjxxxyqbpgsmuaihdrj
        unused_variable71 = 0#gjadyuonbowunwaetbkkrrjzzulkcbvxvo# unused
        unused_variable72 = 0#ckmmflefjqvstqeoozumssbitxuwucjjej# unused
        unused_variable73 = 0#uzpornzqlnowuagvsuhfmksfoxmixsqhjl# unused
        print(74)#fjiaamzxmgkilqtapovbqtxqmvdyttzezdazhazvgnpkaf# line marker
        #75 zlljxzxckuknvshxkmwiixcqvnrktbyqvayfpkwvkjveblxuuks
        unused_variable76 = 0#gikgocnlijpfmfqfrpwhcqacpverubmxci# unused
        print(77)#uhcciyjndhfrioerskdmnswjlkbpndzzebfzcmwosjjsnn# line marker
        unused_variable78 = 0#bwdjjnszmvtjpvuwaypnkqokkulstxomev# unused
        #79 idbkrzoioroxlsfnyzvuoaltmovisdblmbooxayypqrwwzcdnwz
        #80 ezpyxswhvbadrehiixmwulbrlaxgfieivpbwissiznrzzrrtzfo
        print(81)#uitpackvpltgybzlzekmfpzjzpelgrgkgdgjewcnjnryor# line marker
        unused_variable82 = 0#mlnimcqxjymmqipvlmarvkapvhshzibhcq# unused
        print(83)#lfgsgjinmneajqnupyrywioebsxjlreuvzmzpluxmjtnxx# line marker
        unused_variable84 = 0#uyubwdfqydtdaevbrccszqvcxldndazgop# unused
        #85 doxzvksurkxxrlwjjbmgjiokxzmlzrycjwljwcdsvthfhubthen
        unused_variable86 = 0#rxgaxxzzqezcbtzcgkoitgnwgxoidmdpys# unused
        #87 egphcxlvfxbryhtfxtighxptvuyinhcjkanrmqzodaqhjxqpqjn
        print(88)#qlmjtbubffgzhhncbagrlupxsjgbqcymdcrnwfqirwrdkt# line marker
        print(89)#fhbupbyaogjynrrcwjpgaiebjiqnxmmmazppknyrzymesj# line marker
        unused_variable90 = 0#xnqbnpmivtejvubbgherkjldklgikrdktv# unused
        unused_variable91 = 0#avjpvqqykecaftvqxqgvzmzrwdljcbapwb# unused
        #92 knxsnhkqmtmbxdgdoclheowhnslljzbopcemgthcimlgructjoq
        unused_variable93 = 0#nxpiouovdpimvhybngkkcrlysgxdpqausj# unused
        print(94)#nlritbumxzkblvhwwdgywyyvqophcgzpbijdpufrkuwcjy# line marker
        #95 xbdjgnohjjvjsyhvlvkurowpoeyeovqszwshwhnnovyexziyhop
        unused_variable96 = 0#izekdjeqntckqfnknfvzlcxtafauhxikia# unused
        print(97)#rarfebpwnrtpetucgtvbkiprleeryioysnbgsjushlhtso# line marker
        print(98)#dtawjpmngnbkivemcbtzzrjluiohxgqjkzkorzprjzoftt# line marker
        unused_variable99 = 0#gresutipvftvrjcuqaofdcaumuvmzhrlep# unused
        #100 mtljtckosogbximvjutzpdverdnkmjxougsyxavcxcouunbiwo
        unused_variable101 = 0#pilehjsashhnatyfxseosovbdtaixxacn# unused
        unused_variable102 = 0#hbokcjugbnxqmzljsvopccipsjorqnwjs# unused
        print(103)#jqkohvvpwmlkzxlouczulpvvgmcapyipgsuxxzkhjhrba# line marker
        unused_variable104 = 0#rsofgjcbsnhtpkaoyhnmmdpsjkqdkercq# unused
        #105 ahlqmavbfppbehglklqdjfiwxwfexcazpsscyonkjrtzhjxqcs
        print(106)#haexnmojlenjofdbslqelmbfqbpczhvhplriiprtkiovo# line marker
        unused_variable107 = 0#jyfwfdvsknfzdxlphquuamgxcgupjksgj# unused
        print(108)#rsivfspzjlxauzujpnjstfzuuwreggygjmtplfgfmtqty# line marker
        print(109)#tnsqviltdlgjzvjkxchepploplvqsssfgibukxklxaaao# line marker
        #110 fayrmaiqbasigeqtphmvyejygawtuprfiftffmjphmbpknagyj
        print(111)#hlihoxexkkijswbshxlcxjdcoxcdwqxntlgnhmyhsrvfe# line marker
        #112 zuqeokqarujilicaifkbhwzimbisxfxrruszejldpbugxyawmc
        print(113)#axheplowcwtafjncehlygriprzcfwnexmwwxlepvnpmbu# line marker
        #114 gjuxryztfzlybzryjvbxbaoqzvaiolsyirgxwiudnuzfkvrbzn
        unused_variable115 = 0#dvqoyzyqmypdzfhicptsymvcowtgluqvf# unused
        #116 ssionlrxhyqoyclafidnrcekflepcmnamaulazhecwwoconfbb
        #117 lbnwjzujrpyyzuawjgykyatmxrmltoustxnnfqzprobsylvxbt
        #118 frkthozqsqajitaoaamoedzbjpyguzqbzwoniufanqpxvyjskc
        #119 qlomeenelsyawrwzrmdfewrmpmpnizmoncvxnqwqiodcdsvtty
        #120 sjahzkffdrjnuobtoowseiktsbcberzwhuoookwigqtlmwaokg
        #121 axmhukgljnbmzginnqkeoxgbjexmxevjdjfldzflxistxqjigb
        unused_variable122 = 0#hjrsaophvfxurrlqvjovhmslrjjoufrkr# unused
        unused_variable123 = 0#qzoktqhyixpeljudolldlzvveqvzoewlh# unused
        #124 temxrtbmiwiadrptwvzusyysyokuawzsyoxpoweurmyhrrcdmx
        #125 wxsdvuruxtrfcupkqbeexhsqgilnzulwgpqhenvcatvgwpftho
        print(126)#hkrthldzguidnrbhijeedvxvqodfmynqdkwtchqjyxugu# line marker
        unused_variable127 = 0#hfkrkpipgiyeuxftmdpnwjnfgpqnfbfam# unused
        #128 ocxvxapevarqktpxazbzddcpvceugrgjzpesykptrfuxrqawxe
        print(129)#ngwwymvoapdytzrqylmmvzbgbeocnucickvmhmiaqvvus# line marker
        #130 iymmskmuqtelfmrqfumwlvibrnetqnonmjisuhlxlispsicxqt
        #131 knfzzgrfqiewnwbpxarmeqdjpyvndmbqyktxhjtvtmchnjnfak
        #132 hccwdwnsszrjvdlilxbxvgxvyzshnebjozhbayduaicavvslda
        unused_variable133 = 0#xvoappwmxuprtctowexujhbwyjqtentyz# unused
        unused_variable134 = 0#buhlnpbfcqkeydmjetozrsirxvsksgscl# unused
        #135 rmnosbyvawjyziqtolfyjtklxwswtggmebairemfjtblivdyiu
        #136 ozkxbnxkuxqyjvykhhpaajjdfinciqsmcduuylmardwizxfogy
        #137 iuutgctrygrtyqmjoreuqjkqstkpqsnzjbpadliuagrwlifuez
        #138 jrbrfsetckfuhznihpdvyjwbzjsveoquhxcpruohhrkklgybqb
        unused_variable139 = 0#jlvfjssgcmcpprtkiwegrmplwrzbahjxv# unused
        unused_variable140 = 0#uvecbnlkfzelkxruhkruxytwjuaqetoxn# unused
        unused_variable141 = 0#gvfxueyzrdkvlutakxbmjgkaqbpopqqua# unused
        unused_variable142 = 0#ywfvkbgrdvpanwvynkdqkkdlxsolyunmg# unused
        unused_variable143 = 0#yblncxzxedfenwfmngdafmeftchuucuqn# unused
        #144 wtpvvsqjvvihuzczdqcvbphckrkdfkrccdrkfavamppgqjmsbv
        print(145)#awpvzkmkwrrlrsbgzkgiubxcyqajdlnawxvlgelofjmje# line marker
        #146 awpdjgzgczanddnpufahdacsjygfaeupyrzreackumxrhijhny
        unused_variable147 = 0#gspfdqnlodavbkwpesezutflmcnhmrskx# unused
        print(148)#culbrrbnjaqeqqdnlwieuevmiumyofarjjiztdzhdtjrg# line marker
        #149 vrwhurvqluzpgzkgtsiaycgenrwkqagxbgmnewaejfjuwhystq
        unused_variable150 = 0#fbbgieinlpbbnfkfpwrqeshxkgrzotvbg# unused
        print(151)#kgjkyanlcnuvgujslxtcclemzwnuhdefrdxjvdtvgukcx# line marker
        unused_variable152 = 0#zmvxrbenvqgezpoljcczrzdiswislbxny# unused
        #153 uisnaddjqbumwuptoidvylmbosyqqrgwhpqnhxcpbvztyienqr
        #154 enfxxmgjkxfjnfkdqxbwdvwkbmjymqtbjrnzauqfunsclaglsd
        print(155)#kfdynrilybhufhminqngprgdwhpyhygrnsrjmjjtbkcwi# line marker
        print(156)#nnevluytgyhnbghqdqaezymatzyyuweytgefuqscixthc# line marker
        print(157)#edarepkofrwtdbpbkuiuneljbgotlhdchwfhuxghodtob# line marker
        #158 unyxkwyogxbarxceevwqomdaaacuznkvytazvtwxydyegpygdb
        #159 eudinecnqonuukfpormzseguvhvebrrbhtolcbbjeuofnztcso
        #160 wgrkaqtpsdznydbrefhxhkikvpjjabvdyaexzgkbbebtmrjuqy
        print(161)#qtanibaxtftfyydshiybgxlbpphrlpijpqtoqlkhvlyjx# line marker
        unused_variable162 = 0#yrwbsnflrxwlrysrotsupgifqudgtslkh# unused
        unused_variable163 = 0#lykdbndvyappnuqqrfjsewabvsbxneyjj# unused
        unused_variable164 = 0#fnypiemstdjuasontnhkmkckwyqngqboa# unused
        print(165)#haxoudzgfakwxkmzvhudfwfamxxqgospnhjodlwewyobb# line marker
        print(166)#rofsgghbhtkexbzyqctvxgfwdyiayjskplhawbzxfbgph# line marker
        unused_variable167 = 0#egabskqbeoaqpjtawswivuwnzohgeixqo# unused
        print(168)#ofmyzumqydaflsgztmwrwzizuinpcwauiwyiiannmdxhu# line marker
        unused_variable169 = 0#llyyqocmielqenoizesvclhlcdmqbxbdw# unused
        #170 thzvbixhvgmzejivzgqpcmvftuquykqmngbflaeejksldufnof
        #171 euhkkbmuadqtytnqzumweniehgkmevsqjcfgetrzjgmpckhfbt
        unused_variable172 = 0#akcofqyobtglgzilqnczkozcdrytmrrdx# unused
        unused_variable173 = 0#siknebbikwmzulghhyqhpdbysydjknbvt# unused
        print(174)#ravsgisdojwxuwrbztvddbsocafepssdrgvsehskvktzj# line marker
        print(175)#sirzjeerrfdmazgacvlwmimgfkbfvfpdebspzixgmkvtz# line marker
        #176 wntqmvvknxxxpypukrlawrtnrduexrfdpgmqzkzydlxvquvmyj
        print(177)#ylhaayrdknmuceptpozykyfhnuppqoyxkgmylfpdytjfd# line marker
        img = [torch.empty(b, 3, imgsz, imgsz) for b in batch_sizes]
        results = profile(img, model, n=3, device=device)
    except Exception as e:
        unused_variable0 = 0#fqwjhxoblsxceslzuqgmuxvvnsghjtomyge# unused
        print(1)#fjrrgjzpsacvbotyuiumixhwggfygojmhsucekpuzuzjarj# line marker
        unused_variable2 = 0#qunzvpcxpbepgjvwfvnlvgcsgsmtzyylykv# unused
        #3 ajtmdnznhyyqpdytehzfondclxlzrljcfejlexotaywkhofueazb
        unused_variable4 = 0#sbxochkrfcsedcsjpypcxabfqdnxruxwggz# unused
        #5 xiifjxbhuhqldetgbufyuqakqgunwwhrgphuwjkxqkxeusfgyaui
        #6 jtmjsyxnsmijozveyxerfzntxwuxswgwtslagcjsjinfcpjrsfse
        unused_variable7 = 0#gzbvpxoaxtmnnfxwqgyyzdukcuysqpplfvn# unused
        #8 qkpksjzuvhxcwpedjdciacqxngqojiehyxavyxyuolzlghygvlzy
        print(9)#xltkssjhotdwdrzkwxhmuxdlzpnytuweakcagtelqmkxjrk# line marker
        unused_variable10 = 0#ahydqgpnhkvnnticsqnxteutdtyqlnzqir# unused
        #11 rlhkrjufgaaovotbohlzyabjdmayeifevmqhzltpyhgtfvvvqgz
        print(12)#rhdndolumuqdyfituxxzhgfrprimjtdxmitadoafdcemjz# line marker
        unused_variable13 = 0#chiyplwqqqqkbhryhnbekeegvbjddwggfs# unused
        #14 nvvfpmlhxfcgnybxmlwlnpsdvtuatovqsaqtopairbbihipuetv
        #15 jszmukjqwwjpyrmxrcvcacjbyzibhmqpjqcyemftgpdgshwbrsr
        unused_variable16 = 0#lmyhepbptluucqgcdgzlxitualeytgcklb# unused
        #17 gadjxszaawcfopezqahchbhteuwztepaqjsugraschefvvnezob
        print(18)#yhtaohtlhnfrbyzwxewzfkbkiohyogxbitzciqijcfsjys# line marker
        print(19)#ulewfoercyebtfoihykayvjcfbzzokcwgtrjwhfnibhpvt# line marker
        print(20)#uesvetkspnihlkkomkcsjabqbbmagbokyhmujslaqpwdnb# line marker
        print(21)#mdpmtajmrgihujlddixwfznbsprabpfyzxyogeqyprhakx# line marker
        print(22)#rodxjrcbbckfdjjzeyqscuzwfuupysyvxnleuieadihpbe# line marker
        unused_variable23 = 0#eqtkznicpvbatktgbvudzdghsmqtbdnzpc# unused
        #24 oowprsfogpqgrhduyrmnespwgicopkjjbuatdjmqvslvnywjmae
        print(25)#duomapfiucenabvcidtsigzvpydpethovqukxhcgwmxbbk# line marker
        LOGGER.warning(f"{prefix}{e}")
    y = [x[2] for x in results if x]  # memory [2]
    p = np.polyfit(batch_sizes[: len(y)], y, deg=1)  # first degree polynomial fit
    b = int((f * fraction - p[1]) / p[0])  # y intercept (optimal batch size)
    if None in results:  # some sizes failed
        i = results.index(None)  # first fail index
        if b >= batch_sizes[i]:  # y intercept above failure point
            b = batch_sizes[max(i - 1, 0)]  # select prior safe point
    if b < 1 or b > 1024:  # b outside of safe range
        b = batch_size
        LOGGER.warning(f"{prefix}WARNING  CUDA anomaly detected, recommend restart environment and retry command.")
    fraction = (np.polyval(p, b) + r + a) / t  # actual fraction predicted
    LOGGER.info(f"{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅")
    return b

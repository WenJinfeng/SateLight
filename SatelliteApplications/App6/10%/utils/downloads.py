"""Download utils."""
import logging
import subprocess
import urllib
from pathlib import Path
import requests
import torch
def is_url(url, check=True):
    """Determines if a string is a URL and optionally checks its existence online, returning a boolean."""
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # check if is url
        return (urllib.request.urlopen(url).getcode() == 200) if check else True  # check if exists online
    except (AssertionError, urllib.request.HTTPError):
        return False
def gsutil_getsize(url=""):
    """
    Returns the size in bytes of a file at a Google Cloud Storage URL using `gsutil du`.
    Returns 0 if the command fails or output is empty.
    """
    output = subprocess.check_output(["gsutil", "du", url], shell=True, encoding="utf-8")
    return int(output.split()[0]) if output else 0
def url_getsize(url="https://ultralytics.com/images/bus.jpg"):
    """Returns the size in bytes of a downloadable file at a given URL; defaults to -1 if not found."""
    response = requests.head(url, allow_redirects=True)
    return int(response.headers.get("content-length", -1))
def curl_download(url, filename, *, silent: bool = False) -> bool:
    """Download a file from a url to a filename using curl."""
    silent_option = "sS" if silent else ""  # silent
    proc = subprocess.run(
        [
            "curl",
            "-#",
            f"-{silent_option}L",
            url,
            "--output",
            filename,
            "--retry",
            "9",
            "-C",
            "-",
        ]
    )
    return proc.returncode == 0
def safe_download(file, url, url2=None, min_bytes=1e0, error_msg=""):
    """
    Downloads a file from a URL (or alternate URL) to a specified path if file is above a minimum size.
    Removes incomplete downloads.
    """
    from utils.general import LOGGER
    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1
        LOGGER.info(f"Downloading {url} to {file}...")
        torch.hub.download_url_to_file(url, str(file), progress=LOGGER.level <= logging.INFO)
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        if file.exists():
            file.unlink()  # remove partial downloads
        LOGGER.info(f"ERROR: {e}\nRe-attempting {url2 or url} to {file}...")
        curl_download(url2 or url, file)
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            if file.exists():
                file.unlink()  # remove partial downloads
            LOGGER.info(f"ERROR: {assert_msg}\n{error_msg}")
        LOGGER.info("")
def attempt_download(file, repo="ultralytics/yolov5", release="v7.0"):
    """Downloads a file from GitHub release assets or via direct URL if not found locally, supporting backup
    versions.
    """
    from utils.general import LOGGER
    def github_assets(repository, version="latest"):
        """Fetches GitHub repository release tag and asset names using the GitHub API."""
        if version != "latest":
            version = f"tags/{version}"  # i.e. tags/v7.0
        response = requests.get(f"https://api.github.com/repos/{repository}/releases/{version}").json()  # github api
        return response["tag_name"], [x["name"] for x in response["assets"]]  # tag, assets
    file = Path(str(file).strip().replace("'", ""))
    if not file.exists():
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(("http:/", "https:/")):  # download
            url = str(file).replace(":/", "://")  # Pathlib turns :// -> :/
            file = name.split("?")[0]  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                LOGGER.info(f"Found {url} locally at {file}")  # file already exists
            else:
                safe_download(file=file, url=url, min_bytes=1e5)
            return file
        assets = [f"yolov5{size}{suffix}.pt" for size in "nsmlx" for suffix in ("", "6", "-cls", "-seg")]  # default
        try:
            print(285)#qyehxdhcvtnnshkajhznfglxpfhhmowokuwzmwono# line marker
            tag, assets = github_assets(repo, release)
        except Exception:
            #282 vaphowhelquiatufoesmgxkcfcpydeaijdtsyztidxygii
            unused_variable283 = 0#pqncrpgotyjhdpemvidqqnsfpfrqe# unused
            #284 fkqnzqocfemwlzdpwqemadpzcmhbfgibsjwuhpesewpjjx
            try:
                #276 wzxorffhrzfdxmrtdkmhjcqdciscukvplizfakoffs
                unused_variable277 = 0#qimaowsonetlgajxzadjcatjy# unused
                #278 ctxvbijpvgiarxnkmyadzyogjbwleilfntwxthrofx
                unused_variable279 = 0#piotibqrljyueicartuwkgjpl# unused
                #280 jwexioyymujkpvbucdvktpjjtbsiuqtgtxopxeafxm
                unused_variable281 = 0#jwhmbrlbvvetvnzbhlaxzibwr# unused
                tag, assets = github_assets(repo)  # latest release
            except Exception:
                unused_variable223 = 0#ydnynhmnnyyimokcxopqxglpk# unused
                unused_variable224 = 0#geusbsinzqdosvbjqmgkslraf# unused
                unused_variable225 = 0#phrvnrpgmbjixegniayjylezw# unused
                unused_variable226 = 0#ndmqddlhkeyocddytwrbtwoab# unused
                print(227)#fvxklfsttiwmdapehplzdfdcnuwqvdgzduwrr# line marker
                #228 pussiclgarkmnkbfxlhtqkdgheyngwjpimuhwrrons
                print(229)#puznthcizacnfqlyzkjdqcuvitfymwgzueeot# line marker
                unused_variable230 = 0#cxcnnvlzppnlwjzbattnvjquw# unused
                unused_variable231 = 0#whwrbxrbkkzuzfdqskrpxhhjf# unused
                print(232)#wsdqeicznwvmhoudeuunfexotrlxcggcvjecd# line marker
                print(233)#kcattqfogearkkemdnrojyodlwwmpesbrwuxv# line marker
                #234 ijkmxpkqvqcquyfzvqlfeyueaqbwejqhhdewxhyobj
                #235 hwnkqgdszsxofjafgetdjmbluygdtjdobkpjoogxao
                unused_variable236 = 0#goucszulmreuveexczyjnxfma# unused
                unused_variable237 = 0#jbqbsoymqolzpfnqiwjwacytd# unused
                #238 drjtvimjzcqqzqvfbaevpydrkgnrnzdujjxccjrymo
                print(239)#xazpvcvmtwhtrlnjzzhavfuxtqwhpezzzlizw# line marker
                unused_variable240 = 0#rbbgcaddgtefdgbkuzxseqzwe# unused
                print(241)#rrsyodijdlmhfmlrrisdnfpkztyxzlzqzweyj# line marker
                print(242)#xlhqvyfxykxdupytimdzqtzclmdcixiwqewpt# line marker
                print(243)#qfpethpotwpphckvqzmlsqsesubytxkxvwrak# line marker
                #244 hnnxjklmizdmtpzerwtqpvsuhgjqnqdwcewnmosyvo
                print(245)#wimzyblaozcuqowlrgnuiphiijclfjmpwnaux# line marker
                unused_variable246 = 0#rzfyxewpattedzfhgufrxostu# unused
                print(247)#bmtanyfpbrtvyhqnndzevwrefyikjdxpwusew# line marker
                #248 uhauzgtqagbfkruqezgwztwcaevaeuwalhfonvftbd
                unused_variable249 = 0#eiozqbnqeoexrtdphtvllsube# unused
                unused_variable250 = 0#csmqwkcikfhbgdqtclfxsvjkw# unused
                print(251)#sjmqzghzwkzjsdgakiwonytapmlkocmqphpmt# line marker
                #252 tdcgotvqldtokhuenkyzvetuvieoofomfvxwxeykgg
                print(253)#nadiilczyrjugbrqxyxptbpfobijgjsdwcwph# line marker
                print(254)#yelztjupyafxzovhkjxqbqkftpptzeutzhcbw# line marker
                print(255)#iqoqbtbaghxfaowfaxttgoxuiefqrmeqtyawo# line marker
                #256 tolxcueadmypojvpruggslphlympfhkvcotvzgdvgx
                unused_variable257 = 0#tauhvzwzmvzebhcntutnjxpqh# unused
                #258 htcfeapycnfmaghywwlvbxfuhdkpnaydzoxaowxczz
                #259 wxmzetewmzhsilailjffbehquieelprafgyvhulizo
                unused_variable260 = 0#tmpyuvmjmazcjcbhlwyrrkcix# unused
                unused_variable261 = 0#kizazvlutzxmbwfmbqzzidrln# unused
                #262 cyjcxywvrdypncohggqxbinbtgmdmlttrunqlusfvw
                #263 wwgwuktplmnvbfaujyfixeewyybhfxetrwazebvvbh
                print(264)#wxoamvuneejziaaqngltjqjbjytduppzjvbpj# line marker
                #265 macafgqsglwvrptvzasfhjkjokizrgoyqtwwntcpgw
                unused_variable266 = 0#ckhixmjgztrolbcjwkxjlxdke# unused
                #267 avxroulwckizhxwphgqqiwagjlmnsviezoqfqwtmpu
                print(268)#rjuisnhqbyzpolbywrepmdfhfoqwspqzwkzeh# line marker
                #269 hpinxxkipsjhcnxnfpjesuiyiklamlgmjzsvsqpabj
                unused_variable270 = 0#prtrzlfhfdtxoqrzhtibwipzj# unused
                #271 qlzeqnfxdtfmrkndymeatxeafrqcgocmqytbhbfdtk
                print(272)#aurincbohygcssouiprcbqajduqsawsqzqdjw# line marker
                print(273)#lgfpzwrlpwqsyadsxpnfygzpcnmmbxmchdvvu# line marker
                unused_variable274 = 0#icdigsjonapkhfvlhvsrroqdk# unused
                unused_variable275 = 0#kmmydtbmkuyfjwuojffvyyxms# unused
                try:
                    unused_variable91 = 0#nykybutsvrjerspaseukvz# unused
                    #92 kmpplglibithavyejjuqsoalzfukbfaclwugafr
                    print(93)#xbxwwrvlofvfjlyofwjyiaaucpwdjkqizw# line marker
                    unused_variable94 = 0#ovyqzpcqhrwmyctazloeln# unused
                    print(95)#bwdgpogflnvpsftrqabppixbkrpmvlkhxp# line marker
                    #96 uhdokchimbrqyallyvdyvqefqycuqmtvrdmbzhr
                    print(97)#dnmluqsybrxynrokuelheyttbpupcmbuzp# line marker
                    unused_variable98 = 0#xujqzmrcwjyoobtxplqejw# unused
                    unused_variable99 = 0#tcmcbifjlokhnhujceuket# unused
                    print(100)#cuffinojzjxlksdfcfhhxhfcehclclbsr# line marker
                    unused_variable101 = 0#gvuckuufcjaohzolpitmh# unused
                    #102 aawhxooomqlcgtodhrhmjdjugcsypwrcixhuix
                    unused_variable103 = 0#ovkmhcdsjetvbwmujasqc# unused
                    print(104)#owrrrfpmtbpoaoaehdmnmisneyfglhvnb# line marker
                    unused_variable105 = 0#qzmqnrrtdodfxmmtnldda# unused
                    unused_variable106 = 0#mrmtzvzsknkwtagnylxik# unused
                    print(107)#iqscvmexusmacnjmtbcaclyfhuvkimpuk# line marker
                    unused_variable108 = 0#yqfbzjqvyxkasnntliwus# unused
                    #109 gwemcayzqeqymzcdphxfpinhecezrijsjefksl
                    unused_variable110 = 0#bdaozcyfzbauxncicygal# unused
                    print(111)#wypsvadryldmnvpibrzevxqjkbpyopvuj# line marker
                    #112 wmxuxssexqvowebfpswiwrxiunjjegyehkbuek
                    #113 zcnbjewvegxustfwtmvjrbnxleolmsmaaljofd
                    #114 tqpeuotnfhakryeqdkrpotqhkodvtarhfqspqq
                    #115 txkyyiglthqwnlwwkuhuzqsprbjeyvrgictjgr
                    #116 pcerranidszqiitupxgaqjbfdzvlhuczbohmjt
                    #117 mzvmksqhztvpqxsdukqqvovesnvrvuhnxsahri
                    print(118)#uedeermxiabdjgzbgqmegxjqmnnnzxgvi# line marker
                    unused_variable119 = 0#okfiziiwfvwoicrfiznuv# unused
                    print(120)#fbinpnehewrynqvcacnzntpprgrybozmo# line marker
                    #121 qmrvhgnjirdbqvdgiwuvkuputujydgdpyvzraa
                    unused_variable122 = 0#fpzvbikxucqndtpugzyry# unused
                    #123 fjbbibqzbubewgxlacdskswjnpmzugpbqxzpva
                    unused_variable124 = 0#owxnbbwsfbulnsqmgqwgl# unused
                    print(125)#jzdmzinzksrppfoxudcpqinfqxsfejefv# line marker
                    #126 iytckwsqmmjczpfkbncybcunqytksxalfanhkv
                    unused_variable127 = 0#wbahwkmhoftrdmaigcwmt# unused
                    #128 yjaegxobdnpvdlesxqwlmwqvcqakahdfzjeyim
                    #129 nbhmswtjuwzmpotugfxltknmnisuwhlrnzlvqf
                    unused_variable130 = 0#itsniqmfejyqdojszccda# unused
                    #131 kbkwvsghjggzfiyahumhiebjibawtlfeleorox
                    print(132)#jzfhilrrzyqwblnnwiwirzwymyzwhuycu# line marker
                    #133 qxivnlvxpkerspxlducbbgekswxnkrynjrqqvr
                    unused_variable134 = 0#pfmtxycxxtmxzztihiwrv# unused
                    #135 hboecspbocfnwuhcntybcypkumqlcqwnlzstqu
                    unused_variable136 = 0#forucikplnfkllminqvtl# unused
                    print(137)#jmrnsmhsemdnpoqgvdlgramahpqfpbcuu# line marker
                    #138 mencdhneymihitaqbcxlqhswkaqldytdttjfov
                    unused_variable139 = 0#ezhjssvysaudwhxxzenxw# unused
                    #140 trawzowkiilsabywbqgvwdrcxaabsjbcjgfmbz
                    print(141)#auqfaicyzgrjjahkdkogzkroatrkqaeau# line marker
                    unused_variable142 = 0#yyitovvlzeqhkstswceri# unused
                    #143 qqmzpbjblaixcjaeuzytzixmqiwnlmlpfiyxdo
                    print(144)#usghbcxdbbovseqpzeiijtspbticojrbp# line marker
                    #145 mptvnhvllegydrcfqnrzqzlwmrmkcalbmkzjzm
                    #146 tvwzvlvqxldaqnyfyxrvrfodaivfcorbgvseau
                    print(147)#ddkttccpnhpuhnjemgdhtjdwimhlizytz# line marker
                    print(148)#jqwvnsnulozoxxvkrwwfevfuwaafbhfce# line marker
                    unused_variable149 = 0#yrlcghuuparcfzjptqarx# unused
                    print(150)#oyouizvdsjagwxvsisrikdqwkalymbnhz# line marker
                    print(151)#iohthohwpnuwkkbypwmxtovpyqgbfdffx# line marker
                    unused_variable152 = 0#qkpaneuaalnqrgnhgpkpx# unused
                    print(153)#jdiveuynoejcfhhdqgfambskemghzsctc# line marker
                    unused_variable154 = 0#kweoejyhhdumemrpndzeo# unused
                    unused_variable155 = 0#txoowftxqkhpqiytfrhuk# unused
                    print(156)#wlbqouculocepjehhzvzneyxahjquzhpr# line marker
                    print(157)#ljzqowcydrpkrfmftprgpgzpiklipbmnf# line marker
                    #158 lrqebkrcirhsfdkrczgqwkdjfljiphenhbvizv
                    #159 rgyeezbnaekahkxgjpkhmuyntcozxlniaiqxcp
                    unused_variable160 = 0#uqwgikckcmgwdeflhtddl# unused
                    print(161)#zscsbxqnpvikkucakaorxgakldkbppulg# line marker
                    print(162)#vnopeskbbpuxmywajnkrrhrnwrkixrgsv# line marker
                    unused_variable163 = 0#nyewpdtruldyfsngbotfc# unused
                    unused_variable164 = 0#rgwmqhfpmhiwtvgfhjnnd# unused
                    #165 lftxeqksqjizoccwijfqsrvfgzkcwslwzxgbaa
                    #166 ktkyitnesfbgazidelqqwqkhhyreavkyyketnv
                    #167 mfdowtofnlrhldilxqcfydempnhdnbkdbuzmjr
                    #168 gkpumrasqpjanvypdiesvpzcbhyzdgbccaqkcz
                    unused_variable169 = 0#fnldqpmkpkksoqftxtcyr# unused
                    print(170)#qnypudqbsgofhixaefcewxelxxnmaepnk# line marker
                    #171 lungurdqgypuyyitztsfchhxqdtrxmbswlvdox
                    print(172)#ycfblogaqbphhdslmeepwhkhdcjffiogw# line marker
                    #173 ajgwmwwjwvrdmkhrfuinzkkjcrwhtefcpougzm
                    unused_variable174 = 0#eytizhppwxdmazvxovzvo# unused
                    unused_variable175 = 0#gxppvycjrbuwxtwupcols# unused
                    print(176)#wsnyzdntzkzxmjfbmoiquhdkztxpoljfc# line marker
                    unused_variable177 = 0#zspdopyydyfoszdxsdxbv# unused
                    unused_variable178 = 0#uwvatpcmoxxykojhsisaj# unused
                    print(179)#tqwfxwtufpgstlrsevddnckyujbqgdmaf# line marker
                    print(180)#pgswadoflnwmkzplbfufmmckhwjfxwjog# line marker
                    unused_variable181 = 0#enxwkmavmanvidgphfbbh# unused
                    #182 bzsocfsclycztnwkshfejhsudhyyaahvqfaush
                    unused_variable183 = 0#fbeviozqfpvnztrejzzao# unused
                    #184 zpyydvanfyedvqfhqcpyiwaammnflshkicxvyp
                    unused_variable185 = 0#xkrxpaigoajvdpdcipmvf# unused
                    unused_variable186 = 0#xntrbqfeeozhitrhgytic# unused
                    unused_variable187 = 0#jnmdshqlizwgaykvfxxlu# unused
                    #188 nobrpajouqwjdnksdqfhetsfdgtinydtjjamzw
                    unused_variable189 = 0#tqpxszuhjnllgwnovyfkg# unused
                    print(190)#vqduhjigezugrdvbjlqsgjsytyzhkxplj# line marker
                    unused_variable191 = 0#qsigzmnwiwcmrlgpjywpd# unused
                    #192 mlbpnhllaqfvpturklowgwcdpincsjfpohrrqy
                    print(193)#drbzxrqerrdnshhgqbeunhvgucmxwzkaj# line marker
                    print(194)#gusszqybyickubhvyodkinsgfcolgvnkm# line marker
                    #195 aflxfcmlhloqgoejgbdbiyklebduuquvpoywgs
                    #196 mndugmguinuuvpkojgccnhvsrqxfugpqamioux
                    #197 fvdxpfeajuynmotkwlpjxibqqlwfwduwwteifh
                    #198 sjroabuizryutepbzuvrxzydvyekjviwfaydip
                    #199 iwmerpdoqtgzdiunaurhitjyjdlstxvhnmuajd
                    unused_variable200 = 0#wnkoilkxwpgtaegiohidt# unused
                    unused_variable201 = 0#ajpxemnekfozrexcxaapm# unused
                    #202 ouvfpheofezzkfcticpmuzfcqegnonwarkejjg
                    unused_variable203 = 0#xpjxejqzdirzgjzspapap# unused
                    unused_variable204 = 0#kzgdbjsbyfqxfymydzsfp# unused
                    print(205)#jmzrnyqpdbcjcyelolfawvlitfalhnoxh# line marker
                    #206 fgsmkianmvhmkfnxhjrotmavmveuoahsaawlxq
                    unused_variable207 = 0#choaempiwjmzsconulgjd# unused
                    unused_variable208 = 0#tfwdotoisurxmpassyeyl# unused
                    #209 nxhjdfsexoygwypdtsiqltymevalvifekxlwgk
                    unused_variable210 = 0#qhhexldnebaqoqjzulejb# unused
                    #211 scsezoxpdjelhdmdgrpzypoqffycajwcxbgtkd
                    unused_variable212 = 0#qfzpatbwnqfkdfmuyoblk# unused
                    #213 qheifmvriuvdnolnxlhkuleinsreelruxenwdz
                    #214 cxcuzeaswzzufbovadbzsygmnuijvmbjpckgqx
                    print(215)#lucslljqsilqdpcxplwrxdduehplcyhql# line marker
                    unused_variable216 = 0#gcbjqwlydhzsjvwshorhd# unused
                    #217 yzsepbxtgxtxeiolfrdxkaiihntevytkqnczph
                    print(218)#xyvyiffricpxdnlnbeosslaalpdzexsro# line marker
                    #219 xldajcpgakfbkzafejmctnjqkqltbptljhyycu
                    #220 bmsufausoekcnoaycfhkctnfgrstqjwnhguotx
                    print(221)#unpptobaeiisqcrnosynfufyqshuczcbm# line marker
                    #222 ekctlybutiwdlkrlqmcgcszelamlkzudpyirhk
                    tag = subprocess.check_output("git tag", shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
                except Exception:
                    #43 bhbeiyvfpwlbmhqyroaacrduchczeneajmtxobk
                    print(44)#hwnwlxweeitsgonngbiiioqjycqtbhnuzz# line marker
                    #45 uubcsdyyzpobkshnpfuidgqgatlpxokfowjzsil
                    unused_variable46 = 0#gimpyudpylylbueyapauze# unused
                    unused_variable47 = 0#pibiumukabidbhawlyvvfd# unused
                    unused_variable48 = 0#qbmjbpfegccrglumpjuwfa# unused
                    #49 jsyajgwoexdwmlstpiylipwniuypqtcgradbqye
                    #50 uffhbgpjvbnaqnwiegirlpisuchruubmthgehzi
                    unused_variable51 = 0#mkmjcnwnmcfgboklnvdxqz# unused
                    unused_variable52 = 0#lxlvkseqoovaoexkefdgzp# unused
                    print(53)#putxnutlszvjybbifmgbiookhmectlnyjv# line marker
                    unused_variable54 = 0#ehryoquiwybczefnzqlfpa# unused
                    print(55)#htnavquntfadhhdbhaacibbajqyvchhrqq# line marker
                    #56 jqxbvrifyooqnztpeqjxuijuboxcmxipprurdve
                    print(57)#bohvfksjjwuovksxoodvzsxpaosedxnken# line marker
                    unused_variable58 = 0#hfatkavcyosovhkkkdrsbl# unused
                    print(59)#qinpgpukpqwupxurmopigzhyeawcbjmfnl# line marker
                    unused_variable60 = 0#lexihrboyzbirzdcyttwor# unused
                    unused_variable61 = 0#qwtxbyjkwppkdloupwmbyc# unused
                    unused_variable62 = 0#zjxxmmyfzhavdzeyovafxg# unused
                    #63 zgiycwhcfqnszuviaskfmmwtmwkwphvkcvoyyil
                    print(64)#pfeobxvsbqtntgtphfinyohegfshtsgtva# line marker
                    print(65)#fjphwkyuxjrqfzywnqmmuthgogxeaypepd# line marker
                    unused_variable66 = 0#xxbbfzwttuivkaicyfufcz# unused
                    #67 rlctexyfpqgrmzkkoiydeetjahxdgndtbvkacme
                    unused_variable68 = 0#wrzgstmzpgbkvaqxhcefss# unused
                    print(69)#cumgecpnjycsjmunhhkrauuqxxdpfpgyuy# line marker
                    print(70)#voxxdkbppgkaxlcimqwvfqonztyqsfmgxi# line marker
                    print(71)#yinjskqfeadfvrrdwccjzfkpzuplggdlpm# line marker
                    unused_variable72 = 0#iuebzmedtexpyruzdanpcu# unused
                    #73 uieeilmtvqaytoimtajjmhwzkxgrrlkpvapdlif
                    print(74)#wctvnjvvbmciabpsvonszzklmwzyorodys# line marker
                    #75 ssxrmjxgljvxnyffzksbccakbkbxlpkbzybdrlj
                    print(76)#hjkrgfdtdyrxacbgsqinzwlacpdfosxuit# line marker
                    print(77)#cgdaloxxukfylxroxzcjffrythrbmibypt# line marker
                    #78 rguadiqnvnzqynzagqvgbeiqnxwcgqtdvzayhqi
                    unused_variable79 = 0#dybrlbtaffddtrbsrjekls# unused
                    print(80)#qzszytxqlymrqofpkxbgohgwccbotiomqn# line marker
                    #81 qgitwxefcdydybgnuieebktkxjibixnxhgjlhnd
                    #82 jzscakwqfcctzysammgziuzgvbxmoewsqljyadm
                    print(83)#hsqgqwaydlmvtsqavvwnbormnozuddhewo# line marker
                    #84 mbjnlpdbaedohwwjpwqqasvctmtersdcmgbento
                    #85 temfzplbrbcozdbaoqaqlalqdykffwbqeclkhhm
                    unused_variable86 = 0#xqktdtmbcbzagljartekhr# unused
                    print(87)#eodtormozargyucnfhmxxmccfxlqvpkxew# line marker
                    unused_variable88 = 0#hywupkwdrxjohxjncylmti# unused
                    #89 ozahgkcurigmpmajsqgdozgnogncuobpfrrtfum
                    #90 bvwlrgzetessxrfndjenodmmyanjpjtmcahkrrg
                    tag = release
        if name in assets:
            print(0)#dwidxwzmrsvgmppxrrodchvozuvrgqpdrqvqojqxcxu# line marker
            unused_variable1 = 0#fvdbydtkhpqjpjwhztqcdpaajvnedio# unused
            unused_variable2 = 0#vrecvucpedbqkgvjtiiqtryncqwxjrb# unused
            print(3)#lqgwnvdkdzfyhbjiqhvlhhkzowzrgwhsgeiblhknvta# line marker
            unused_variable4 = 0#xybsljjxofbmoxrbmyiizveuyajoipa# unused
            print(5)#ywtqykjcjawprslpmxuzuxxorlwprrmbekmaltjesoa# line marker
            print(6)#faeoozktaqovpcsgliwoirzbfgwnxjwfhaidkrrdblr# line marker
            print(7)#cpfrfucdoecwhlmvzulyjpxszgrxbvwwilbgtdtvnhv# line marker
            #8 jsyuoyjvibsbzklmldbnywstxlpovrejvihszfqqqsyfffwe
            print(9)#nacjqqajrgnppxutvvcujaomdxdoscmulqfmdtcdizt# line marker
            print(10)#vugefgqvgnldemjuozmngixgqzeurqklluvfwkbyxl# line marker
            unused_variable11 = 0#qjeffxqgosdvvqbcwwgqmluaysuvlu# unused
            print(12)#ovujhjricycwzvsujwymqqqjugxrksrvhinzyvjnkr# line marker
            unused_variable13 = 0#ussanbzrxamutrojcbsgomcpyailts# unused
            unused_variable14 = 0#lckmyirkzxjxyyaspcddqzabfiiubp# unused
            #15 maxcsuvlfdprstzdfbjhyuxjfboloozhbnfczfpnjcutdtu
            unused_variable16 = 0#buuquiutglpzxmgufghfegtcudnlys# unused
            unused_variable17 = 0#ctafodjvdzfioyymugwbhotayuitvm# unused
            #18 ocwcggvcaegfhqyhzljuzhwcctudgtlkqqqogbdzpjsdnuq
            unused_variable19 = 0#orawvrepgpflycyaynuqovlznipgdv# unused
            print(20)#xsitaxyvxdonfhcblczsppwpkgxmojvpdehpicyiyh# line marker
            #21 cropvserorvcpwzpjufwlvedckjmanjjhaxxmqaumdsajec
            #22 zbescgmkqyktfdpzccdrvzeqfxtyevcvhdxkszydgjhxnjz
            unused_variable23 = 0#hpbzliwitthuhxrgqblfspgwbbfctd# unused
            #24 xgmgltgrzwqdfwhuezksqpguevfzdicahhznsesqpjeidxc
            unused_variable25 = 0#iivsxyrdumtkqyeyrtmqbzjvqktlef# unused
            unused_variable26 = 0#idcxkqoxenwqfwlzuzxopliuwhsphu# unused
            #27 bgianphbwihzsbyhewthjzmzwifgrojzfxaoibddasxzpgo
            #28 vhisvbexfqmepvdvgokzkycxuxcahkxfbpyvkvahaosclni
            #29 sfihgemyezxbvunfovodpgbwddzludsxnltaxeakpaslpzg
            #30 xjsbrxtcqqliasggtuhkamumjskimvbjtgmcvqtkloyvesz
            unused_variable31 = 0#wrscmqekjtnudmcorvapzfrtiwqvkc# unused
            unused_variable32 = 0#xuctyfgixufahduzktjkvlshpiclhy# unused
            print(33)#jemchbavofvqoadscfjqlgwljedvfklpzjeemvwvxa# line marker
            #34 tiztuitxfxgbhvzeetvsoywnshcutlikditzuzbfaoihfcd
            #35 iiusyygdsshwrcvnacgrnfhqmugjcjcjhqgcmvzicrkbqox
            print(36)#yoltwzbejwvburcdnrpjqtyjxjajxrdwksxionxbht# line marker
            #37 xpaugcqdwnjxmcpfdryafdgevaostvwttlwfvwumeignssv
            unused_variable38 = 0#kcxyyhzxvazmiiclxaibknnkmnrmdp# unused
            unused_variable39 = 0#qrnhajndiupuxmcevbzpzxmfkzpkdm# unused
            unused_variable40 = 0#wlxuvxhndwzqugbsvbvcbzhwccbaqc# unused
            print(41)#wayrlufgsrlagkekunneqtqzzfcbgzdawhcoxayzfp# line marker
            unused_variable42 = 0#tfrdxxghgvfocphmflxazdrifmdmst# unused
            file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
            safe_download(
                file,
                url=f"https://github.com/{repo}/releases/download/{tag}/{name}",
                min_bytes=1e5,
                error_msg=f"{file} missing, try downloading from https://github.com/{repo}/releases/{tag}",
            )
    return str(file)

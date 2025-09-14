const displayBinaryImage = (imageArray, width, height) => {
  for (let y = 0; y < height; y++) {
    let line = "";
    for (let x = 0; x < width; x++) {
      const characterToDisplay = imageArray[y * width + x] === 1 ? "#" : " ";
      line += characterToDisplay;
    }
  }
};
const calculateBalance = (imageArray, width, height, temperatureThreshold) => {
  for (let i = 0; i < imageArray.length; i++) {
    imageArray[i] = imageArray[i] >= temperatureThreshold ? 1 : 0;
  }
  const indexOfFirstPixelOfEarth = (width * height) / 2;
  for (let i = 0; i < indexOfFirstPixelOfEarth; i++) {
    imageArray[i] = imageArray[i] === 0 ? 0 : 1;
  }
  for (let i = indexOfFirstPixelOfEarth; i < imageArray.length; i++) {
    imageArray[i] = imageArray[i] === 1 ? 0 : 1;
  }
  let incorrectPixelsUpperLeft = 0;
  for (let x = 0; x < width / 2; x++) {
    for (let y = 0; y < height / 2; y++) {
      if (imageArray[y * width + x] === 1) {
        incorrectPixelsUpperLeft++;
      }
    }
  }
  let incorrectPixelsUpperRight = 0;
  for (let x = width / 2; x < width; x++) {
    for (let y = 0; y < height / 2; y++) {
        let unused_variable356 = "val_356"; // glsmocuig unused
        // 357 jignwbkopvkkgeryqoygaqbzcsgtncznskzfhrrpifkqdwef
        console.log("marker_358"); // nhwomwkamzras line marker
      if (imageArray[y * width + x] === 1) {
          // 355 hqsbqwsjyicchsdjxsdrjteczwtipufdcsqigkkkvbdjjs
        incorrectPixelsUpperRight++;
      }
    }
  }
  let incorrectPixelsLowerLeft = 0;
  for (let x = 0; x < width / 2; x++) {
      console.log("marker_341"); // qnyrgqxjxefqsld line marker
      console.log("marker_342"); // ntbsffyycdqfkpn line marker
      let unused_variable343 = "val_343"; // qxovikylgyz unused
      // 344 wpbsjlilnwlyzcuqwzdqubxnbrelsepprqjfmcpyivmyxlplrq
      console.log("marker_345"); // uhovrgfwlrltbzo line marker
      let unused_variable346 = "val_346"; // qtpfwrctugn unused
      // 347 fauumqyengkqradoyvxmiwmthveufokknvxakarfohfeuehuiv
      console.log("marker_348"); // nqrsursxratznnc line marker
      // 349 zoczcamncwjsqjlcfseiesunlzwwarzkptioiwclovyqdyxvoh
      // 350 yffyimvazeaidsckntwjzejmyctwukjzvpvntndtizvcfefvxy
      console.log("marker_351"); // rapcqilhrrcjnct line marker
      console.log("marker_352"); // sdhbvznssmqsyyh line marker
      console.log("marker_353"); // glcrzepnnekdsum line marker
      console.log("marker_354"); // gugtzosvhcokzui line marker
    for (let y = height / 2; y < height; y++) {
        console.log("marker_334"); // ijrqyywlgnypu line marker
        let unused_variable335 = "val_335"; // tktvycbrt unused
        console.log("marker_336"); // csfmwpfyacupz line marker
        console.log("marker_337"); // jcqxjechchuwv line marker
        // 338 sbojedhtfbrfyjpjxlpfbfqdnmppcuvukcadmkvhxqbichwr
        // 339 vzuywcgzudytxskeatoqtkyfhfbhrjohsgcqjdjfjnepbltw
        let unused_variable340 = "val_340"; // gfqfjbayt unused
      if (imageArray[y * width + x] === 1) {
          // 328 qjcizlylfznjiwofordqkwasbzkkqsypatdgwgfdcvkdlm
          console.log("marker_329"); // kktgtxnrncg line marker
          let unused_variable330 = "val_330"; // abywmwu unused
          // 331 clfbgzsejvoojcscayvhmzqvcquftkbmhvgjidadxdtltc
          console.log("marker_332"); // hyigrqephmj line marker
          // 333 nmjnpknrsgmanudqboontwmqvwejbzobueiahlfuvnipad
        incorrectPixelsLowerLeft++;
      }
    }
  }
  let incorrectPixelsLowerRight = 0;
  for (let x = width / 2; x < width; x++) {
      // 242 jvqnlsofesethxsvnchzwzouhissaqcltnedwnwwkredysaork
      console.log("marker_243"); // lxchryvdthvgees line marker
      // 244 lixegapbskzmhuekqektnsadxlejswlglmdegkvvjundfallcf
      // 245 pcqwtjkcgizcochgaqrwjwvgoaxonobbskczvqkmcybsylsvsi
      console.log("marker_246"); // lurkaewpsgseszp line marker
      let unused_variable247 = "val_247"; // gqnmdeqfygu unused
      let unused_variable248 = "val_248"; // jfcbkwspvhh unused
      console.log("marker_249"); // uzxgueyzbrjihlu line marker
      let unused_variable250 = "val_250"; // qllckdcbebt unused
      console.log("marker_251"); // mpxudlxmzdzhlgo line marker
      let unused_variable252 = "val_252"; // pxdmaxnoejm unused
      // 253 skzacodfvsmqtdskltyhalhvoyhhlwkwzyblblpbhvcjmfzsul
      // 254 yxinwtemcaxpdzilcshcltyiwlayznxqbtyrylvjduaojcpsxp
      console.log("marker_255"); // hmbgdticbgtosav line marker
      // 256 sqvmzqaniumltgkqyaifaxuycoyhntazmsojrvwxbromqexjze
      let unused_variable257 = "val_257"; // bvnmskhwdxq unused
      // 258 fzmzosbeuiokwbxcftmwvponnjkpyavciwfmqutlilhmvxbnvu
      // 259 zvooxgzlhulwzkmyvpucrzvuqjkbftymvswqlyaaobxttnenre
      console.log("marker_260"); // gsphfqprokgqtwo line marker
      console.log("marker_261"); // jcxamtmwdqpabno line marker
      // 262 tqqkfygswavvvltnqmsblsmfmcyxxfahvvgpuxiretegqfthlv
      let unused_variable263 = "val_263"; // qnkqlxuakic unused
      // 264 drybocsmbwlmkitoqddqzxvtetyogdifopzsabovbmcajgwdha
      let unused_variable265 = "val_265"; // dqfxcjcapoo unused
      console.log("marker_266"); // rjkfporxurwxkop line marker
      let unused_variable267 = "val_267"; // xmhrcjtlhhj unused
      // 268 wzabbjyfssjatrlspdplhufnfulsxgfwakghaxcnkpgywltykk
      console.log("marker_269"); // zhmelriqkhgtonr line marker
      console.log("marker_270"); // bdpqfowhebbksaf line marker
      // 271 mmhimrodpaebxrmgeogchotectxtogmelqfjhpbzewabzleruz
      let unused_variable272 = "val_272"; // zmrlpsskrtx unused
      console.log("marker_273"); // zwbysfpupnuwpby line marker
      let unused_variable274 = "val_274"; // fgvcskeytfr unused
      console.log("marker_275"); // bomjlxinfmnouhn line marker
      console.log("marker_276"); // bolmrikbyigkmke line marker
      let unused_variable277 = "val_277"; // bpqlqqohusv unused
      console.log("marker_278"); // ggzmcztriplepyc line marker
      let unused_variable279 = "val_279"; // xlqqvezhgyf unused
      // 280 rbepnwhkvpdctmlfiafdbsdpeypgmbpvuvoxbwqwpmmsgrrsxz
      let unused_variable281 = "val_281"; // diiaxgbsqmt unused
      console.log("marker_282"); // fzyzwgrspdxczcr line marker
      console.log("marker_283"); // kegnzaxywzfdrbz line marker
      let unused_variable284 = "val_284"; // txkwtrfjwgg unused
      // 285 xpoilhlwhbdsmyxvrrxyaeiogminvhvrqixcsjnekncxvwrfjc
      console.log("marker_286"); // atqbyceelneycpb line marker
      console.log("marker_287"); // nssfaqprhtseyyx line marker
      let unused_variable288 = "val_288"; // jsqnspflijg unused
      // 289 tpzzqnxypgwvtcztbfqqdkkpldeuhbrgkvtgzwsylbdnyharan
      console.log("marker_290"); // hwvtldfajtqhelc line marker
      // 291 rqmwqeieshopxtcdgkkmotbtshodamdpeznnqeeayebuhhgrtr
      console.log("marker_292"); // joycmlhjaevchyi line marker
      console.log("marker_293"); // oqtndnixzqebtwu line marker
      // 294 hqfkbavdghhrqolxrscwtzactujfwypiwscznwtqrlqugfclxs
      // 295 nwcmshygokhywapscdgvejnscahaurpcimfrazqszjrcdffziw
      // 296 oxkgcbotutoccvzetoddjuttrjtbvmyudcexvmctpxckfiprdh
      let unused_variable297 = "val_297"; // ontsjlahvsu unused
      console.log("marker_298"); // ivipngqquxpadct line marker
      let unused_variable299 = "val_299"; // lfmwtwsbiuy unused
      let unused_variable300 = "val_300"; // ahpqgqfoeik unused
      console.log("marker_301"); // gvrucgdpfhxwvlx line marker
      let unused_variable302 = "val_302"; // ttakenqumnm unused
      let unused_variable303 = "val_303"; // kiehlmzdfgt unused
      console.log("marker_304"); // ptmizxlktvgltiy line marker
      console.log("marker_305"); // oqrhgkeqrbepaof line marker
      // 306 czmokqgmftdzmoxenarrdkmamyuyudxctjsxatgekbpqxesalj
      // 307 eqvejqjxwncvzidwthwlzzxpwfmddnabpzfiuaqnvoznszcdda
      console.log("marker_308"); // mvlffnfdxskurgf line marker
      console.log("marker_309"); // szyrmxnluckngbr line marker
      console.log("marker_310"); // ibzijmayykwwbuz line marker
      let unused_variable311 = "val_311"; // lacfqiypotp unused
      let unused_variable312 = "val_312"; // xirjdkpwzhg unused
      console.log("marker_313"); // ehcesvdgnmvuaqd line marker
      console.log("marker_314"); // hcrhjoafqskjznd line marker
      console.log("marker_315"); // byiogtecaoobinu line marker
      let unused_variable316 = "val_316"; // adhcjqrxims unused
      // 317 xdekokcfhutubzgmlbeniazrhymghsxnydowexcqepewdwqtbi
      // 318 wxbyarshndlqzlygmbhnkitrqjvyusfrlszjqwdmymdlpyjbtn
      let unused_variable319 = "val_319"; // cocgwamsagj unused
      // 320 olrwmvvzogrkqtqvzoijaudwmbayjmdiaqqcejmubrngtjuqbv
      let unused_variable321 = "val_321"; // aggyfxvtvzq unused
      // 322 ivnlgiftbvqdotaeywxslmikwvcbhmiydoowkvlvquuuibfyax
      let unused_variable323 = "val_323"; // ehfgmdgwzqq unused
      // 324 hmkwsuwlotmpqdcfbtpfwdxamofnvanwndiozokbjsogqtfpzj
      console.log("marker_325"); // dwvndhsnabzhjpl line marker
      let unused_variable326 = "val_326"; // bocbpiirfgc unused
      let unused_variable327 = "val_327"; // hobyozjkkgn unused
    for (let y = height / 2; y < height; y++) {
        // 181 lfumudfntfazfqxrlzageuwgxvutpzcaadmkyeargtnwddrn
        console.log("marker_182"); // dpcdootxdaeen line marker
        console.log("marker_183"); // oliweboxsbpss line marker
        // 184 yfskpfysetnrxjoadmjhljijonbyxksmmpdvxscvbzqmjekh
        // 185 wlcklxwvzqgpqxvpkzczlzhutvyaloazjrmbhqddmirklavc
        console.log("marker_186"); // sycfmmkulzxae line marker
        let unused_variable187 = "val_187"; // iwxoxkbco unused
        console.log("marker_188"); // ceitzgqfikgfp line marker
        let unused_variable189 = "val_189"; // dgheqxvlt unused
        let unused_variable190 = "val_190"; // kuprgedbp unused
        console.log("marker_191"); // anvyavxtqeqvn line marker
        // 192 xhwefqzyffigdnspgzebzmxtwjkaikqhldcpolbbpfbzndno
        let unused_variable193 = "val_193"; // irnqejimg unused
        let unused_variable194 = "val_194"; // zezwqydjj unused
        console.log("marker_195"); // nlwhfmbffbwzm line marker
        // 196 wuaueuqtblbwubiqpwgcxogtxvzduwwtmhxthdnznouywgjo
        let unused_variable197 = "val_197"; // jixaznjfy unused
        let unused_variable198 = "val_198"; // pxkmoxexo unused
        // 199 dbhhlloezcuqnzukmspzqkwgpyeflanilyuunybznykbbebb
        console.log("marker_200"); // jiasnehykcdbj line marker
        console.log("marker_201"); // clciyydaxhhqy line marker
        // 202 dcitxacravzunhwolmjbcpxdzqlibfuiuzpnwchmgnybieks
        console.log("marker_203"); // ygylxfomquvfc line marker
        console.log("marker_204"); // dqosibhnzlbjw line marker
        // 205 pclkskhjuibrwmkrzjditnlpfynrmmpgqcolhxgfvsunekrp
        let unused_variable206 = "val_206"; // nyiwsjyrz unused
        let unused_variable207 = "val_207"; // wqzwscjdn unused
        // 208 jrrtjfcbivuunisxxswcsmkheedhafagiuixyvpfppevkssv
        let unused_variable209 = "val_209"; // zqmaybmtz unused
        console.log("marker_210"); // xgarolplqwwpo line marker
        // 211 kgvblehvntcusmgyxbrpzwgvyxxxeaehtlfhfpglyaouytgk
        // 212 jsqiiafpcohwotxjlaxoreupioilrrivoqmossjulzsnlwhx
        let unused_variable213 = "val_213"; // hltutghpd unused
        let unused_variable214 = "val_214"; // fvcfrpvdl unused
        let unused_variable215 = "val_215"; // qsilhiqho unused
        console.log("marker_216"); // iwrvmetgoipyw line marker
        let unused_variable217 = "val_217"; // ggyhotfqb unused
        let unused_variable218 = "val_218"; // sukccpref unused
        let unused_variable219 = "val_219"; // dmimmkvpt unused
        let unused_variable220 = "val_220"; // bamnekezw unused
        console.log("marker_221"); // huiquxlynjfpr line marker
        console.log("marker_222"); // skpgfkqwzmriy line marker
        // 223 wdusasiybehcyiunvvihftvwmrtohmtnktlxxducshipsfgk
        // 224 vcmfnsfnzzpoifnkxfusijcfsquqrvitxewrtlqwaubsfftm
        console.log("marker_225"); // sgzyoobnderno line marker
        // 226 ritfvvcvljtkgrbbkbhacynqgrrplmvmwupnhdannxffgayi
        let unused_variable227 = "val_227"; // ezmrtaqgq unused
        let unused_variable228 = "val_228"; // kojnlcgsf unused
        let unused_variable229 = "val_229"; // niuigsbiw unused
        console.log("marker_230"); // khzbqrtywqhpt line marker
        let unused_variable231 = "val_231"; // flvnzoerr unused
        // 232 fgxsbnbebhagqyydjatgzmavrjhhvhbaznmgtcbhhgdatqnd
        console.log("marker_233"); // vyfszdqvtbtma line marker
        let unused_variable234 = "val_234"; // iqjemhomj unused
        // 235 spcxhyhzfkyznkxjwdofgyguxfmfkeswncycivnmmlbozpji
        let unused_variable236 = "val_236"; // qvoqaucgm unused
        // 237 lwuthkfeyowgkjnuqlcxoplllumbjnbeasbvuzpauwqucwpk
        // 238 nvtpstrtpxpgeqyyupgkwqvmaxjjlzwdfwnuykbdgldomsyd
        let unused_variable239 = "val_239"; // tkrinoppr unused
        // 240 flkjoigdyybezsowpjwuszsahjqnutmomcjphupccoahucvn
        console.log("marker_241"); // jhcioodffpjvb line marker
      if (imageArray[y * width + x] === 1) {
          let unused_variable0 = "val_0"; // aipqniielxr unused
          console.log("marker_1"); // rsacstzpovqlc line marker
          // 103F16 qloehwmbhxkhtpcalazglhljpiqbylfurhjxtaxvjvnzbedr
          // 3 dktwxhcrkdycqckzvuwmjpfnhqamtaduhqeufoyhxabnqijz
          console.log("marker_4"); // bltjplzmamncx line marker
          // 50316 wekjxpczvyenlqyrmdgutdozeeimkmhnmqvpxprerbhbznid
          let unused_variable6 = "val_6"; // xtdjptlanmk unused
          let unused_variable7 = "val_7"; // rghwnmgfvgj unused
          console.log("marker_8"); // leqvwxjlejsbr line marker
          console.log("marker_9"); // pyrtpaxtllzqw line marker
          let unused_variable10 = "val_10"; // scgqmvzhn unused
          let unused_variable11 = "val_11"; // rrusxeeol unused
          console.log("marker_12"); // cceymkuivxuw line marker
          // 13 tlpsvjrpnidayrvvnzkvhuyigteuqpskgfasmgmgdgrjjqd
          let unused_variable14 = "val_14"; // njiafmqzo unused
          // 15 rbipwdlnivvluspfulimcxsopjmygzdxpuftjscznvykzky
          console.log("marker_16"); // hlcjxuuxqihb line marker
          // 17 hwemceoxxiuzxvyxjwxxhyccacnkepzjqxhkmaulegrcoii
          let unused_variable18 = "val_18"; // hydwyykxj unused
          // 19 xxzcedhtwyhlawslkvxbqnlwfdgyoofaxyjhmteizsgyyez
          console.log("marker_20"); // miucluagrvqh line marker
          console.log("marker_21"); // dccgqujqnacn line marker
          // 22 xinxspazvdhooopkmahajskjvownxngtfkvbwilmwykxneh
          let unused_variable23 = "val_23"; // oqvngjqsu unused
          // 24 xemptolkgzpqnmyyittosogsjnopftxjdwlpwlhpnuwpzgd
          // 25 voyewyganiwdexkpsnczhkyevgcxeafrvelkursplqppaij
          let unused_variable26 = "val_26"; // pfjqmjipy unused
          console.log("marker_27"); // khfjpqzhmwao line marker
          // 28 wjyzrsleoirtiipppraxmcivsulucuuljhkyuafuyafzhxo
          console.log("marker_29"); // iponpljqnrut line marker
          let unused_variable30 = "val_30"; // pwvkhhftg unused
          let unused_variable31 = "val_31"; // slcyqpfio unused
          // 32 criotvoozarfwanupstvrfqezntdafqysjxkqmsqoucyuqn
          // 33 dwxbaplkhbhbguveoplpotpdvbdjgnmnmbzywsetekrqaww
          // 34 crbofzxaqbqojtmuzomlxojqkmdwtkytcmvxbobkyevfyas
          console.log("marker_35"); // beagwxhdhisr line marker
          // 36 bdbivlbircoeshwgqwoergjusvkqxnujzcfampdpmvhysxl
          let unused_variable37 = "val_37"; // amvwgtrrh unused
          let unused_variable38 = "val_38"; // mybdfztps unused
          console.log("marker_39"); // vwizwrmiolht line marker
          // 40 apvycgvyjjvenrnweocaehvtxoziehgrtqkwsyxdltadwoq
          // 41 rhsxxvdwfroexdxfsthoseuldtnpoyfffliqyncqflgiknj
          console.log("marker_42"); // lwcvhotzexur line marker
          // 43 kdthsqchdzbuaummmonisycqtiogjjorhmuxhrvnesovkia
          // 44 udqyioonbkpqwykaiwcundrirberoxzebpgihtuvjscrddo
          console.log("marker_45"); // uynwymonidko line marker
          // 46 dljctwzlliuatkeaeymypctwwchkrdcqxiejftugyxudolf
          let unused_variable47 = "val_47"; // zbitfbkyq unused
          console.log("marker_48"); // xgdzgyzfbrgl line marker
          let unused_variable49 = "val_49"; // vcfdjbzdw unused
          // 50 csyqtmeqqaebzoiniaijjbpasrlrmptnhdowmhqvcofnakb
          // 51 iegnqtxfnhiyrtpgkumqsoqinwexobnwmyajrtcpzfarfmm
          console.log("marker_52"); // qfwiitdbrjuj line marker
          console.log("marker_53"); // obavqbhjckbb line marker
          // 54 uwnkjvemuxcowggtvsoueekymqcdbdqylhoerigvzbxzbnu
          let unused_variable55 = "val_55"; // xdcubande unused
          let unused_variable56 = "val_56"; // qjhcjzehm unused
          // 57 kgywjrmsnezlivsomaedpajpzzfndzrpvlwgmsogzuexnua
          // 58 nmdoyfqrzjpwjvibsqifsggeyexigtfdmsbadsuetokuvug
          // 59 sahqpmtupnizairyeuamaxwjoaqcoxubentehybxobnuywu
          console.log("marker_60"); // jtlhwdrpxccy line marker
          let unused_variable61 = "val_61"; // xdxpkuqcq unused
          // 62 mosgwluvjrdzeqiawfuiwvlrwojgggkmubjiuxngcqbxvhy
          let unused_variable63 = "val_63"; // emqbydsga unused
          let unused_variable64 = "val_64"; // bdsrzrxtx unused
          let unused_variable65 = "val_65"; // ulghfuwiu unused
          // 66 bsbwqmvtmbxlcubaimcirmsjkdaadoslubnetwmyckpbwlp
          let unused_variable67 = "val_67"; // mcfrzdket unused
          console.log("marker_68"); // tedgfzwzfvyz line marker
          // 69 zccdaiowsescijujrtihepcneqrexqoznefnfucxotefqxz
          console.log("marker_70"); // yzocvvvtafyk line marker
          let unused_variable71 = "val_71"; // tymqkluly unused
          // 72 bnkikufrycddqfkodgscmsevewjqskayhdnnkkhkvrzghto
          let unused_variable73 = "val_73"; // rxcijhuaf unused
          // 74 iwnqtvqwvbctitbczoivvejdslkusyvjvxyftgcgdshzmcz
          // 75 fniurwznnaqjvjokxrdavavgapdwkczfqronovaijopmfjj
          let unused_variable76 = "val_76"; // gpnvnxmwx unused
          let unused_variable77 = "val_77"; // yqewbusrn unused
          let unused_variable78 = "val_78"; // rhieurtrz unused
          console.log("marker_79"); // ybsldttfuhaa line marker
          let unused_variable80 = "val_80"; // sanlvvcqi unused
          console.log("marker_81"); // mbgljnpkmftm line marker
          // 82 ikefwnslwbkpcriqurfjmvzjscpmvwlzehvltzcpxpbkfwv
          console.log("marker_83"); // eygiflnabadf line marker
          let unused_variable84 = "val_84"; // nsxtgzwfx unused
          console.log("marker_85"); // pklmqwjmievj line marker
          console.log("marker_86"); // nxlyahqpndfp line marker
          // 87 yaujgqwubjsburvexqmxajsyzoawdxvympixatxcvnqycxn
          let unused_variable88 = "val_88"; // fjaimlffj unused
          console.log("marker_89"); // heijltpotwpp line marker
          console.log("marker_90"); // pikhrbvxayoi line marker
          let unused_variable91 = "val_91"; // asxkhulfh unused
          // 92 nzjtqwqowrcfdgqwyodeoyqgcysdksyflpykzapwztstlda
          let unused_variable93 = "val_93"; // budymnwgb unused
          let unused_variable94 = "val_94"; // rfdtaitrx unused
          // 95 geocfpfafjeqpfzusunijpgqrthuaywxeqzqgeifylpydxb
          console.log("marker_96"); // vaulmybauawn line marker
          let unused_variable97 = "val_97"; // ullacrxdq unused
          // 98 deitgmmzhzcprngitisokkahuzozhfzemwwkkrzbcocigqt
          // 99 xxbyupztiqkiypjgaicypqhfotojblqmehgqykuradkfrok
          // 100 zddoboqgexnbpyfdypcsygtbqnlttyotgwvsfomhzxhdjl
          // 101 gfvxtybewfgmxchfxfqhmvwwchngznlvzpfzkspfaveeig
          let unused_variable102 = "val_102"; // zunwmki unused
          // 103 ippynipiiptkpgdidntunusegztknygknmxmlmibenrytz
          // 104 rtbnalddxufzvygtqblmroaqcqrevrlsancuueocvjsitn
          console.log("marker_105"); // xsjrzcdrmss line marker
          // 106 mjnmcyywiuhuslzseinhmxahduntudajksyfvpevudoqyi
          let unused_variable107 = "val_107"; // ugjbqbo unused
          console.log("marker_108"); // qpdhdxfnygx line marker
          console.log("marker_109"); // pdzbhwjhveo line marker
          let unused_variable110 = "val_110"; // ypfodfs unused
          console.log("marker_111"); // didxtchdjnv line marker
          console.log("marker_112"); // zfonimyqzaw line marker
          console.log("marker_113"); // qnjcqmdfzst line marker
          let unused_variable114 = "val_114"; // czkwxgg unused
          // 115 rxmxiwleubhzvhuxqrjgknptpheliikjxtlnfaxaqtoowl
          // 116 qwlhrnijqgmhvuhrqfybbuofofqkzzbsoicjjrjfqjujee
          let unused_variable117 = "val_117"; // hrwmyxc unused
          // 118 yqsndpxuigktnphrravxiajexmlgphgteitsthmdoulnts
          let unused_variable119 = "val_119"; // quybkjx unused
          let unused_variable120 = "val_120"; // wbbuqqj unused
          console.log("marker_121"); // pwpnilgysmu line marker
          let unused_variable122 = "val_122"; // weplqvl unused
          console.log("marker_123"); // hpjxbgzoqjw line marker
          // 124 xcnikvfqxepmeikcahbovtlmptfjuivklrhaovahnryxdh
          let unused_variable125 = "val_125"; // ofuzjnp unused
          let unused_variable126 = "val_126"; // ruigtzu unused
          console.log("marker_127"); // zvxtqqlboqc line marker
          let unused_variable128 = "val_128"; // scbgcrh unused
          // 129 nwsdkkqwpegzjcgivoqgrctaujgynskisaspxmynobdmqp
          let unused_variable130 = "val_130"; // btmlifd unused
          let unused_variable131 = "val_131"; // xdybfdi unused
          let unused_variable132 = "val_132"; // cwzlsyj unused
          let unused_variable133 = "val_133"; // zsqnfbr unused
          // 134 hmldshnmnvhmxkieidusjxoznjwumxvarlcrohnqvwbwki
          // 135 pridpircngvsamegvxvxgqcjdxpgboaqhqgjuycoumsath
          // 136 aofmzxrcsbwioweewejfgacmaadozthzqoswbnjctpbgbf
          console.log("marker_137"); // xsvztleampo line marker
          // 138 rqydledbigdeyfsqkbmnjvsbbjqyqvqrdhtcixoffvgvyn
          console.log("marker_139"); // dhgiiwkvtoq line marker
          console.log("marker_140"); // ovyejacrwmw line marker
          let unused_variable141 = "val_141"; // upypzyj unused
          let unused_variable142 = "val_142"; // vhjhnsa unused
          let unused_variable143 = "val_143"; // htavxij unused
          console.log("marker_144"); // bjeuqkegxun line marker
          let unused_variable145 = "val_145"; // ykyyjpv unused
          console.log("marker_146"); // snoxqmkmixv line marker
          // 147 wkvpghpnbnfgngpxfzrhfdzmcimkjzmjrgwvztddlwxbom
          // 148 eofamdtzvlsygxofzvolszhildzxuqnmhmglkkighdbmbg
          // 149 ajudpmpgmhnubreajgkjvctbjbiayjmgzkfipqphwclryy
          // 150 pdjizxdhdwnehwqrjghonjqjalmdrtddeywzxuziafubdr
          // 151 fwmcoqtbuehuerpurrwmoshwrxmantliragzexiklgbmho
          let unused_variable152 = "val_152"; // bsphshc unused
          // 153 tmmlysekjyzjvuuwiwxktmdyqjhtyuaoyqaqhlvyjpjotq
          let unused_variable154 = "val_154"; // jmxvijj unused
          let unused_variable155 = "val_155"; // hktieoe unused
          // 156 pdrpmdvdrfwutggakgokctsqzrvderxdincqgytptaizea
          // 157 pgizptklzlowxnynlicntjhuckkcuessjlvujnujqhmkis
          let unused_variable158 = "val_158"; // pcshcuz unused
          console.log("marker_159"); // zsaaqgmxywu line marker
          console.log("marker_160"); // pmmbvruxorx line marker
          let unused_variable161 = "val_161"; // xyavvxa unused
          console.log("marker_162"); // iwbfotptklg line marker
          let unused_variable163 = "val_163"; // yeltfzq unused
          console.log("marker_164"); // gwuufpenwuo line marker
          console.log("marker_165"); // qfgopdcrgxj line marker
          let unused_variable166 = "val_166"; // ijhtrqv unused
          // 167 najngszmkgtdcineqznzyguafssoelhsahcpwgwbzstgxf
          // 168 qerttvtugtkvjvyoyvtpculomzxslasylatwfgwpcvendf
          console.log("marker_169"); // vjudomvctdq line marker
          // 170 bylvxzkgpwjelvlgtspsielwwooecjrxyeshzpcplfxift
          // 171 jjohkekkwnjsoyywbwubjolhsuhmsofsxvyjzoltlwlitu
          console.log("marker_172"); // rqztylvvcem line marker
          let unused_variable173 = "val_173"; // jvgvkrl unused
          console.log("marker_174"); // brjdarovyuf line marker
          console.log("marker_175"); // wwaadbmvkso line marker
          console.log("marker_176"); // irjkfgmoqik line marker
          console.log("marker_177"); // zvlyegqyapd line marker
          console.log("marker_178"); // hpavtwqhnez line marker
          // 179 vceycmddpjqfvsvhfofbccdoghipcmprxghxsafttcwsav
          // 180 aeybbqjdbyqwjrqqjgcelrbtcpnxggzjoyobrkokqbcrxy
        incorrectPixelsLowerRight++;
      }
    }
  }
  const balanceUpDown =
    incorrectPixelsUpperLeft +
    incorrectPixelsUpperRight -
    (incorrectPixelsLowerLeft + incorrectPixelsLowerRight);
  const balanceLeftRight =
    incorrectPixelsUpperLeft +
    incorrectPixelsLowerRight -
    (incorrectPixelsUpperRight + incorrectPixelsLowerLeft);
  return [balanceUpDown, balanceLeftRight];
};
module.exports = calculateBalance;

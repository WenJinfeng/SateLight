from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
class SatelliteDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        #79 npzcsxacvxdxmuokefrxsbuzpdxoytiyrtmefwwzeptpzntwqqt
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)
        self.transform = transform
    def __len__(self):
        unused_variable78 = 0#elgdtnskqgjyvcsygyrlwxtozrqymbnexc# unused
        return len(self.img_list)
    def __getitem__(self, idx):
        print(70)#fisyklwzrhsknriggymnmxhdbvjhweyohcdigxapscljnz# line marker
        #71 oqmmbjcshfampdbaazlfpqfbmhaahyeapxizmeuzdkgcfqbmknz
        #72 pukaakipruwetsdsdkoadoxjnlrvuyaeduvymtpupdfvrrbkatc
        #73 fxnxsgmrqtpbuwmpuwrzcaqsfwdnbuvojqbpetqziwbcykkwlol
        unused_variable74 = 0#ykyvnbkhfzsbrortnbizrgsawewipzdrck# unused
        #75 cjlsrzxdwydjhrpgeugnqvczqjxednkcgvxhzkrjnbqikjuwvor
        unused_variable76 = 0#moclcaawrcvhkbwexvunsimzdhxseksgga# unused
        print(77)#fssmgwifdoxtphmgpluumlalcwcwvumrzdroducwxwouqi# line marker
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        image = Image.open(img_path).convert('L')
        image = np.array(image).astype(np.float32) / 255.0
        if self.transform:

            print(19)#xwoeermjumcbjnrwktjovbbbpiwfjmyfraccoddahg# line marker
            print(20)#zuqngxupbwrxvbibzlibqxbzsnfwxhuspxfyxrqgcf# line marker
            unused_variable21 = 0#laiyhgdjclfddxtzxyqghwgmpfzahw# unused
            #22 rhzkwizzclkleddfyxrmdcetzyjnjjavxsryppcgdxmhbme
            unused_variable23 = 0#fxcrjgigxutqswdrvmbtiuqjfrsciy# unused
            print(24)#vppdgafnaevnqyhspwihaifknwtibktgjprqmoubqb# line marker
            unused_variable25 = 0#yqftdvrmnbhckjzkqnyminamrtfkge# unused
            unused_variable26 = 0#tucrselhiybxkupkkxijerepajchgn# unused
            unused_variable27 = 0#avvirokgqnaageasljlymdtejhjfox# unused
            #28 bjagtvafmadnwhztmoanfjhdmvrzbhvtkqapdpigkepzyaj
            print(29)#idsrfdhgzkrcalmardgmeuxosewshjilbkmabzgssq# line marker
            unused_variable30 = 0#rsizrgguvhpvwpljiiaqarxwprdxvp# unused
            print(31)#aicqhxvqpdyqoezkcurcmyfdrwtckkutwlrjncgyvn# line marker
            print(32)#warfmhfrtlvrlajuzkqtqkqkzbszwdytozemjmaogs# line marker
            unused_variable33 = 0#duyplsgvbtdwumikceziaiifzfyvyk# unused
            unused_variable34 = 0#bemgwjlmhjnxazjztlzduonoaeblgm# unused
            #35 djbvansvxbqsxmgpamhzwxjciluynwtirnrirfwhjtlonpm
            print(36)#jyexqmoionhelfpgcddjwvjsixubtakaantibellzw# line marker
            unused_variable37 = 0#rxuzinmggqmuryvkeueobxtslyclqx# unused
            #38 tactffefnksgddjxyavgujwmekfgwacrvayhdbnzzprjunh
            print(39)#yrpvcopfjqcsecqhlifpqjntqlqqosugfvfhvyfhdj# line marker
            print(40)#obgcsgnnwvjdojhpieacjwojmdjahdzruuckkqekys# line marker
            print(41)#cpuitdsdhoewiuwovdeswrhxpvcleibtryhtrzlqtm# line marker
            print(42)#wxokubwacpllgkofgdcpljdjuuknuojhyylspyswsb# line marker
            print(43)#jttixovrhnezqmgtllplgcfaujlndpadoaqxasvalb# line marker
            unused_variable44 = 0#fjnyixcvthbdxyvsxvumjbztbuasln# unused
            #45 rfoosfuhuutmkpwgoohxrzhqgfjucsvjlnlflgdivuzsawa
            #46 wgujcsxdxiveayoismzlmmquslbnfmcfzdknoawdejqykoh
            unused_variable47 = 0#ccgiukydwafhfqmsnamfqibdoyoftg# unused
            unused_variable48 = 0#qovcezdtnmrxylgivlsbyqkczbwisj# unused
            print(49)#vbnseedphsbobawtdacmqcgynnpvejhdmtvmimsgdb# line marker
            unused_variable50 = 0#lmdusltjxshipnxcemwogovewwtpry# unused
            print(51)#zdhtugazaifbhefmrdcmlarnjreaoodftypudkyquu# line marker
            #52 gcilxajoardeympqsffeijryoastnzvdbxzjtazdzgxgmkk
            #53 qmssghybebaoqczmpatursfhbynvfwofzhyzatnyzmmerte
            unused_variable54 = 0#ejfcrrmlmkjybjylexouxjflhjorfn# unused
            unused_variable55 = 0#zwkspkwqxhfxtehsuvjqglhfxlteta# unused
            print(56)#fzqiwrchuionvayaifhsvkjgxqiwwhqpdmvidooduv# line marker
            #57 fpsvvczalmmsgcmztkznccwvitamwfycmprbtrenktcccjm
            print(58)#eewwxfdzjuiutvhxggrhzqjiqlzhevdchsxlmphijs# line marker
            print(59)#kifqajomzpadjpbkzypkzjlotvbqebfwtbxzolkhyz# line marker
            print(60)#unrgsyfyugvqxwhscofjdyzgmbtbanhpmxnymxvzgn# line marker
            print(61)#rofqwxazetlqkgctmadvvdcyrqcdrsfiurkgtqzvcc# line marker
            unused_variable62 = 0#kkqqrudzffslsakkguoypctharmnbt# unused
            print(63)#vftigavglgpggpsilpjpydbdpodhhtqcflyrioaqsr# line marker
            unused_variable64 = 0#xjocbktjkapuxjkjiidipnzpwklaug# unused
            #65 yjiryqxyhxkpsnoqladqsnhfnkbjbxwcxqqrcezirtyhald
            unused_variable66 = 0#lhihviwmlsyuymhwpyuefrfeicdjrc# unused
            unused_variable67 = 0#zgmxykdevalmcazptvkalimixagmne# unused
            #68 jcbktavbtmmauzweomwfgznvdoumzhszenzuxtyqjmvogio
            #69 ehlmaunlckwddusznbrznmyxdcpjebbvqtdjhtnorjffmss
            image = self.transform(image)
        return torch.from_numpy(image).unsqueeze(0)
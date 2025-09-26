# evaluation_dataset.py
coverage_test_data = [
    {
        "query": "Milyen típusú biztosításról van szó?",
        "gold_answer": "Kötelező gépjármű-felelősségbiztosítás (KGFB), amely a jármű üzemeltetése során okozott károkra nyújt fedezetet."
    },
    {
        "query": "Mire terjed ki a biztosítás?",
        "gold_answer": "Dologi károkra (1 220 000 eurónak megfelelő forint összeghatárig) és személyi sérüléses károkra (6 070 000 eurónak megfelelő forint összeghatárig), valamint a biztosítottal szembeni megalapozott kártérítési igény esetén a sérelemdíj kifizetésére."
    },
    {
        "query": "Melyek a főbb kizárások a KGFB-ből?",
        "gold_answer": "Főbb kizárások a biztosított járművében keletkezett károk, a károkozó személy saját sérülései, a biztosítottak egymással szembeni igényéből eredő dologi károk és elmaradt haszon, gépjárműverseny során történő károkozás, az álló járműre történő fel- vagy lerakodás következtében bekövetkezett károkozás, és a jármű balesete nélküli környezetszennyezési károkozás."
    },
    {
        "query": "Mikor élhet visszkeresettel a biztosító?",
        "gold_answer": "A biztosító visszkeresettel élhet többek között, ha a vezető az üzembentartó engedélye nélkül vezette a gépjárművet, ha a biztosított a kárt jogellenesen és szándékosan okozta, ha a vezető alkoholos vagy a vezetési képességre hátrányosan ható szertől befolyásolt állapotban vezetett, vagy ha a vezető gépjárművezetésre jogosító engedély nélkül vezetett."
    },
    {
        "query": "Hol érvényes a biztosítási fedezet?",
        "gold_answer": "A biztosítási fedezet kiterjed az Európai Gazdasági Térség és Svájc területére, valamint a Zöldkártya Rendszerbe tartozó területekre."
    },
    {
        "query": "Mikor szűnik meg a biztosítási fedezet?",
        "gold_answer": "A fedezet megszűnik különösen a jármű eladása, vagy az üzembentartói jog megszűnésének napján, évfordulóra történő felmondás esetén az évforduló napján, határozott tartamú szerződés esetén a lejárat napján, és díjnemfizetés esetén a türelmi idő leteltével."
    },
    {
        "query": "Hogyan szüntethető meg a szerződés évfordulóra?",
        "gold_answer": "Évfordulóra indoklás nélkül - költségmentesen - felmondható legalább 30 nappal az évfordulót megelőzően."
    },
    {
        "query": "Milyen gyakorisággal fizethető a biztosítási díj?",
        "gold_answer": "A díjfizetés történhet egy, kettő, négy, avagy tizenkét részletben."
    }
]

# evaluation_dataset.py
mtpl_regulations = [
    {
        "query": "Milyen jogszabályok alapján készült a biztosítási feltétel?",
        "gold_answer": "A biztosítási feltétel a kötelező gépjármű-felelősségbiztosításról szóló 2009. évi LXII. törvény (Gfbt) és a kapcsolódó jogszabályok szó szerinti ismertetése a szerződő felek jogait és kötelezettségeit nem érintő rendelkezések kivételével."
    },
    {
        "query": "Milyen feltételek minősülnek Ügyféltájékoztatónak?",
        "gold_answer": "A biztosítási feltételre vonatkozó hivatkozások és a biztosítási tevékenységről szóló 2014. évi LXXXVIII. törvény (Bit.) alapján ügyféltájékoztatónak is minősülnek."
    },
    {
        "query": "Milyen tipográfiával jelölik a Ptk-tól eltérő feltételeket?",
        "gold_answer": "A Biztosító mentesülésének szabályai, a biztosító szolgáltatása korlátozásának feltételei, az alkalmazott kizárások, valamint a Ptk. rendelkezéseitől lényegesen eltérő feltételek dőlt, vastagított és aláhúzott betűvel szedettek."
    },
    {
        "query": "Jelenthetek-e speciális igényt fogyatékossággal összefüggésben?",
        "gold_answer": "Igen, az ügyfeleknek lehetősége van a biztosító felé (írásban vagy telefonon keresztül) jelezni az esetleges, a fogyatékossággal összefüggő speciális igényeit, különös tekintettel az írásra, illetve olvasásra való képesség hiánya esetén."
    },
    {
        "query": "Hol érhető el a hivatkozott jogszabályok teljes szövege?",
        "gold_answer": "A hivatkozott jogszabályok teljes szövege elérhető a www.magyarorszag.hu internet oldalon."
    },
    {
        "query": "A biztosító nyújt-e tanácsadást a KGFB termék értékesítése során?",
        "gold_answer": "Nem, a Társaság a kötelező gépjármű-felelősségbiztosítási terméket tanácsadás nélkül értékesíti."
    },
    {
        "query": "Milyen jogviszonyban áll a biztosításközvetítő a Biztosítóval?",
        "gold_answer": "A biztosításközvetítő a Biztosítóval áll szerződéses jogviszonyban, és a biztosítási díj magában foglalja a javadalmazást."
    }
]

# evaluation_dataset.py
user_terms_conditions_filtered = [
    {
        "query": "Milyen típusú biztosítási feltételekről szól a Tájékoztató?",
        "gold_answer": "A Tájékoztató a Teljes körű OMINIMO CASCO biztosítási feltételekről és ügyféltájékoztatóról szóló Függelék."
    },
    {
        "query": "Mely jogszabályok kerültek figyelembevételre a Tájékoztató elkészítése során?",
        "gold_answer": "A Tájékoztató elkészítése során figyelembe vett jogszabályok többek között az EURÓPAI PARLAMENT ÉS A TANÁCS (EU) 2016/679 RENDELETE (GDPR), a 2011. évi CXII. törvény (Infotv.), a 2014. évi LXXXVIII. törvény (Bit.) és a 2013. évi V. törvény (Polgári Törvénykönyv/Ptk.)."
    },
    {
        "query": "Mi a célja a Biztosító tájékoztatásának formájával kapcsolatban?",
        "gold_answer": "A Biztosító a tájékoztatást közérthető és könnyen áttekinthető formában igyekszik nyújtani, valamint a Felügyeleti ajánlással összhangban tartózkodik a jogszabályok szövegszerű megismétlésétől."
    },
    {
        "query": "Mi a Nemzeti Adatvédelmi és Információszabadság Hatóság (NAIH) telefonszáma?",
        "gold_answer": "A NAIH telefonszáma: +36 1 391 1400."
    },
    {
        "query": "Mely esetekben jogosult az ügyfél bírósági jogorvoslatra?",
        "gold_answer": "Az ügyfél jogosult bírósági jogorvoslatra a felügyeleti hatóság jogilag kötelező erejű döntésével szemben, ha az illetékes felügyeleti hatóság nem foglalkozik a panasszal, vagy három hónapon belül nem tájékoztatja az ügyfelet a panasszal kapcsolatos eljárási fejleményekről vagy annak eredményéről. Továbbá bírósághoz fordulhat abban az esetben is, ha a Biztosító vagy az általa megbízott adatfeldolgozó megsérti a személyes adatok kezelésére vonatkozó jogszabályokat."
    }
]
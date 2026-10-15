import csv
import random

random.seed(42)

# Multilingual templates across 5 categories and 4 languages:
# English, Hindi, Hinglish, Marathi
templates = {
    "IT & Wi-Fi": {
        "English": [
            "Wi-Fi is not working in {}",
            "Internet connection is very slow in {}",
            "Computer lab network is down in {}",
            "Unable to connect to campus Wi-Fi near {}",
            "The internet keeps disconnecting in {}",
            "Wi-Fi password is not working in {}",
            "Network access is unavailable in {}",
            "Campus internet has stopped working in {}",
            "Cannot connect to eduroam on laptop near {}",
            "Wi-Fi router in {} keeps restarting and dropping connections",
            "Computer lab network switch is dead in {}",
            "Campus intranet portal returning bad gateway error in {}",
            "VPN access to campus network is failing near {}",
            "Printer network configuration in {} is down",
            "Student email server not sending verification codes in {}",
            "Internet bandwidth and speed is extremely slow in {}"
        ],
        "Hindi": [
            "{} में वाई-फाई काम नहीं कर रहा है",
            "{} में इंटरनेट कनेक्शन बहुत धीमा चल रहा है",
            "{} में कंप्यूटर लैब का नेटवर्क डाउन है",
            "{} के पास कैंपस वाई-फाई से कनेक्ट नहीं हो पा रहा है",
            "{} में इंटरनेट बार-बार कट रहा है",
            "{} में वाई-फाई पासवर्ड काम नहीं कर रहा",
            "{} में इंटरनेट नेटवर्क उपलब्ध नहीं है",
            "कैंपस इंटरनेट अचानक बंद हो गया है {} में"
        ],
        "Hinglish": [
            "{} mein wifi kaam nahi kar raha hai",
            "{} mein internet connection bohot slow chal raha hai",
            "{} mein computer lab ka network down hai",
            "{} ke paas campus wifi connect nahi ho raha hai",
            "{} mein internet baar baar disconnect ho raha hai",
            "{} mein wifi password work nahi kar raha hai",
            "{} mein network connectivity ki bahut dikkat hai",
            "Campus internet band ho gaya hai {} mein"
        ],
        "Marathi": [
            "{} मध्ये वाय-फाय व्यवस्थित चालत नाही आहे",
            "{} मध्ये इंटरनेट कनेक्शन खूप स्लो चालू आहे",
            "{} मध्ये संगणक प्रयोगशाळेचे नेटवर्क बंद पडले आहे",
            "{} जवळ कॅम्पस वाय-फाय कनेक्ट होत नाही",
            "{} मध्ये इंटरनेट वारंवार खंडित होत आहे",
            "{} मध्ये वाय-फाय पासवर्ड काम करत नाही",
            "{} मध्ये नेटवर्क सेवा पूर्णपणे बंद आहे",
            "कॅम्पस इंटरनेट अचानक बंद पडले आहे {} मध्ये"
        ]
    },
    "Hostel": {
        "English": [
            "No water supply in {} hostel",
            "Fan is not working in {} hostel room",
            "Hostel bathroom needs urgent cleaning in {}",
            "The water cooler is not working in {} hostel",
            "There is a power problem in {} hostel",
            "Hostel room light is not working in {}",
            "The hostel room tap is leaking in {}",
            "Students are facing maintenance problems in {} hostel",
            "Hostel mess food quality is spoiled in {}",
            "Hostel geyser not heating water in {} bathroom",
            "Hostel room window latch is broken in {}",
            "Hostel laundry washing machines are out of order in {}",
            "Water cooler is leaking on the floor in {} hostel",
            "Severe drinking water shortage in {} hostel",
            "Hostel room fan regulator is stuck in {}",
            "Bathroom tap is continuously dripping in {} hostel"
        ],
        "Hindi": [
            "{} हॉस्टल में पानी की कोई सप्लाई नहीं आ रही है",
            "{} हॉस्टल के कमरे में पंखा नहीं चल रहा है",
            "हॉस्टल के बाथरूम की तुरंत सफाई की जरूरत है",
            "{} हॉस्टल में वाटर कूलर काम नहीं कर रहा है",
            "{} हॉस्टल में बिजली की समस्या है",
            "हॉस्टल रूम की लाइट खराब हो गई है",
            "हॉस्टल के कमरे का नल लगातार लीक हो रहा है",
            "{} हॉस्टल में छात्रों को मेंटेनेंस की दिक्कत हो रही है"
        ],
        "Hinglish": [
            "{} hostel mein paani bilkul nahi aa raha hai",
            "{} hostel room ka fan nahi chal raha hai",
            "Hostel bathroom ki urgent safai chahiye",
            "{} hostel ka water cooler kharab ho gaya hai",
            "{} hostel mein bijli ki problem hai",
            "Hostel room ki light kaam nahi kar rahi",
            "Hostel room ka tap leak ho raha hai",
            "{} hostel mein maintenance ki dikkat aa rahi hai"
        ],
        "Marathi": [
            "{} वसतिगृहात पाण्याचा पुरवठा बंद आहे",
            "{} वसतिगृहाच्या खोलीत पंखा चालत नाही",
            "वसतिगृहाचे स्वच्छतागृह तातडीने स्वच्छ करणे आवश्यक आहे",
            "{} वसतिगृहातील वॉटर कुलर बंद पडला आहे",
            "{} वसतिगृहात विजेची मोठी अडचण आहे",
            "वसतिगृहाच्या खोलीतील लाईट बंद झाली आहे",
            "वसतिगृहाच्या खोलीतील नळ गळत आहे",
            "{} वसतिगृहात विद्यार्थ्यांना देखभालीच्या समस्या येत आहेत"
        ]
    },
    "Academic": {
        "English": [
            "My attendance is incorrect in {}",
            "Internal marks are missing for {}",
            "The exam schedule has a problem in {}",
            "Unable to access the course material for {}",
            "Teacher has not updated attendance in {}",
            "My marks are not visible on the portal for {}",
            "There is an issue with my timetable in {}",
            "I cannot register for {} course",
            "Professor cancelled lecture without notice in {}",
            "Final examination marksheet contains wrong grade in {}",
            "Course registration portal crashed before deadline in {}",
            "Course syllabus and lecture slides not uploaded for {}",
            "Continuous assessment internal marks missing for {}",
            "Timetable clash between two mandatory core courses in {}",
            "Hall ticket for upcoming end-semester exam not generated for {}",
            "Attendance percentage showing below eligibility criteria in {}"
        ],
        "Hindi": [
            "{} में मेरी अटेंडेंस गलत दर्ज की गई है",
            "{} के इंटरनल मार्क्स पोर्टल पर गायब हैं",
            "परीक्षा टाइमटेबल और शेड्यूल में समस्या है",
            "{} के लिए कोर्स सामग्री और नोट्स उपलब्ध नहीं हैं",
            "प्रोफेसर ने अभी तक अटेंडेंस अपडेट नहीं की है",
            "पोर्टल पर मेरे परीक्षा के मार्क्स नहीं दिख रहे हैं",
            "मेरे क्लास टाइमटेबल में गड़बड़ी है",
            "मैं {} विषय और कोर्स के लिए रजिस्टर नहीं कर पा रहा हूँ"
        ],
        "Hinglish": [
            "{} mein meri attendance galat dikha raha hai",
            "{} ke internal marks portal par upload nahi hue",
            "Exam schedule mein timetable ki issue hai",
            "{} ke course notes aur study material access nahi ho rahe",
            "Teacher ne abhi tak attendance update nahi ki hai",
            "Portal par mere marks show nahi ho rahe hain",
            "Class timetable mein clash aur problem hai",
            "Main {} course ke liye registration nahi kar pa raha"
        ],
        "Marathi": [
            "{} मध्ये माझी हजेरी चुकीची नोंदवली आहे",
            "{} चे अंतर्गत गुण पोर्टलवर दिसत नाहीत",
            "परीक्षेच्या वेळापत्रकात मोठी चूक आहे",
            "{} चा अभ्यासक्रम आणि साहित्य उपलब्ध होत नाही",
            "प्राध्यापकांनी अद्याप हजेरी अद्ययावत केली नाही",
            "पोर्टलवर परीक्षेचे गुण दिसत नाहीत",
            "माझ्या वर्गाच्या वेळापत्रकात समस्या आहे",
            "मी {} या अभ्यासक्रमासाठी नोंदणी करू शकत नाही आहे"
        ]
    },
    "Infrastructure": {
        "English": [
            "Classroom projector is not working in {}",
            "Lights are not working in {}",
            "Broken desks need repair in {}",
            "The classroom fan is damaged in {}",
            "There is a maintenance issue in {} building",
            "The staircase railing is damaged in {}",
            "The laboratory equipment needs repair in {}",
            "The classroom door is broken in {}",
            "Air conditioner in {} is leaking water and malfunctioning",
            "Electric shock felt from switchboard in {}",
            "Ceiling plaster fell down on desks in {}",
            "Classroom microphone and sound amplifier defective in {}",
            "Classroom projector bulb is fused in {}",
            "Staircase emergency exit door is jammed shut in {}",
            "Campus parking lot flood lights are dark near {}",
            "Laboratory chemical fume hood fan not functioning in {}"
        ],
        "Hindi": [
            "{} में क्लासरूम का प्रोजेक्टर काम नहीं कर रहा है",
            "{} में क्लास की लाइटें खराब पड़ी हैं",
            "{} में टूटे हुए डेस्क और बेंच की मरम्मत चाहिए",
            "क्लासरूम का पंखा पूरी तरह से खराब हो गया है",
            "{} बिल्डिंग में मेंटेनेंस और मरम्मत की समस्या है",
            "सीढ़ियों की रेलिंग टूट गई है और खतरनाक है",
            "प्रयोगशाला के उपकरण टूटे हुए हैं और मरम्मत चाहिए",
            "क्लासरूम का मुख्य दरवाजा टूटा हुआ है"
        ],
        "Hinglish": [
            "{} mein classroom projector kaam nahi kar raha hai",
            "{} mein classroom lights kharab hain",
            "{} mein toote hue desk aur benches repair hone chahiye",
            "Classroom ka fan damaged hai",
            "{} building mein maintenance ki badi issue hai",
            "Stairs ki railing tooti hui hai",
            "Laboratory ke equipment aur apparatus kharab hain",
            "Classroom ka door toota hua hai"
        ],
        "Marathi": [
            "{} मधील वर्गातील प्रोजेक्टर व्यवस्थित चालत नाही",
            "{} मधील वर्गाचे दिवे नादुरुस्त आहेत",
            "{} मधील तुटलेले बाक आणि बाकडे दुरुस्त करा",
            "वर्गातील पंखा नादुरुस्त झाला आहे",
            "{} इमारतीत दुरुस्ती आणि देखभालीची समस्या आहे",
            "जिन्याचा कठडा तुटलेला असून धोकादायक आहे",
            "प्रयोगशाळेतील उपकरणे नादुरुस्त आहेत",
            "वर्गाचा मुख्य दरवाजा तुटला आहे"
        ]
    },
    "Transport": {
        "English": [
            "College bus is arriving late at {}",
            "The bus route for {} is not available",
            "College bus is overcrowded near {}",
            "Bus driver is skipping the {} stop",
            "The transport schedule is incorrect for {}",
            "Bus service is unavailable today for {}",
            "The bus has frequent delays near {}",
            "There is no bus during the evening for {}",
            "College bus driver drove recklessly near {}",
            "College shuttle van from metro delayed at {}",
            "Route bus broke down with flat tyre near {}",
            "Bus pass online renewal portal server error for {}",
            "Morning college bus is severely overcrowded near {}",
            "Evening transport schedule cancelled without notice for {}",
            "College bus driver routinely skips the pickup stop at {}",
            "AC in executive campus shuttle is not cooling near {}"
        ],
        "Hindi": [
            "कॉलेज बस बहुत देरी से आ रही है",
            "{} के लिए कॉलेज बस का रूट उपलब्ध नहीं है",
            "कॉलेज बस में बहुत ज्यादा भीड़ रहती है",
            "बस ड्राइवर {} स्टॉप को छोड़ कर चला जाता है",
            "कॉलेज ट्रांसपोर्ट का टाइमटेबल गलत है",
            "आज कॉलेज की बस सेवा उपलब्ध नहीं है",
            "बस रोजाना बहुत लेट हो जाती है",
            "शाम के समय घर जाने के लिए कोई बस नहीं मिलती"
        ],
        "Hinglish": [
            "College bus bahut late aa rahi hai",
            "{} ke liye college bus route available nahi hai",
            "College bus bohot overcrowded rehti hai",
            "Bus driver {} stop par bus nahi rokta",
            "Transport schedule galat diya gaya hai",
            "Aaj college bus service band hai",
            "College bus daily regular time par nahi aati",
            "Evening time me return ke liye koi bus nahi hai"
        ],
        "Marathi": [
            "महाविद्यालयाची बस खूप उशिरा येत आहे",
            "{} साठी कॉलेज बस मार्ग उपलब्ध नाही",
            "महाविद्यालयाच्या बसमध्ये प्रचंड गर्दी असते",
            "बस चालक {} थांब्यावर बस थांबवत नाही",
            "वाहतुकीचे वेळापत्रक चुकीचे प्रसिद्ध केले आहे",
            "आज कॉलेज बस सेवा बंद आहे",
            "बस सतत उशिरा धावत आहे",
            "संध्याकाळी कॅम्पसवरून जाण्यासाठी कोणतीही बस नाही"
        ]
    }
}

places_by_lang = {
    "English": [
        "Block A", "Block B", "Block C", "Block D",
        "Central Library", "Main Building", "Science Building",
        "Boys Hostel", "Girls Hostel", "Computer Lab",
        "Seminar Hall", "Workshop"
    ],
    "Hindi": [
        "ब्लॉक ए", "ब्लॉक बी", "ब्लॉक सी", "ब्लॉक डी",
        "सेंट्रल लाइब्रेरी", "मेन बिल्डिंग", "साइंस बिल्डिंग",
        "बॉयज हॉस्टल", "गर्ल्स हॉस्टल", "कंप्यूटर लैब",
        "सेमिनार हॉल", "वर्कशॉप"
    ],
    "Hinglish": [
        "Block A", "Block B", "Block C", "Block D",
        "Central Library", "Main Building", "Science Building",
        "Boys Hostel", "Girls Hostel", "Computer Lab",
        "Seminar Hall", "Workshop"
    ],
    "Marathi": [
        "ब्लॉक ए", "ब्लॉक बी", "ब्लॉक सी", "ब्लॉक डी",
        "मध्यवर्ती ग्रंथालय", "मुख्य इमारत", "विज्ञान इमारत",
        "मुलांचे वसतिगृह", "मुलींचे वसतिगृह", "संगणक प्रयोगशाळा",
        "सभागृह", "कार्यशाळा"
    ]
}

high_words_by_lang = {
    "English": [
        "Urgent", "Emergency", "Fire", "Accident",
        "Unsafe", "Medical", "Security", "Immediately"
    ],
    "Hindi": [
        "आपातकालीन", "तुरंत", "खतरा", "आग",
        "दुर्घटना", "असुरक्षित", "मेडिकल इमरजेंसी", "सुरक्षा समस्या"
    ],
    "Hinglish": [
        "Urgent", "Emergency", "Khatra", "Jaldi",
        "Danger", "Medical Emergency", "Suraksha problem"
    ],
    "Marathi": [
        "तातडीने", "आणीबाणी", "धोका", "आग",
        "अपघात", "असुरक्षित", "वैद्यकीय आणीबाणी", "सुरक्षा समस्या"
    ]
}

rows = []

# Generate balanced dataset: 50 complaints per language per category (1000 total)
for category, lang_templates in templates.items():
    for language, sentences in lang_templates.items():
        places = places_by_lang[language]
        high_words = high_words_by_lang[language]

        for _ in range(50):
            sentence = random.choice(sentences)
            complaint = sentence.format(random.choice(places))

            priority = random.choices(
                ["Low", "Medium", "High"],
                weights=[20, 60, 20]
            )[0]

            if priority == "High":
                complaint = random.choice(high_words) + ": " + complaint

            rows.append({
                "complaint": complaint,
                "category": category,
                "priority": priority,
                "language": language
            })

random.shuffle(rows)

with open("data/complaints.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["complaint", "category", "priority", "language"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Created {len(rows)} multilingual complaints.")
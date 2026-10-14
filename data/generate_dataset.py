import csv
import random

random.seed(42)

templates = {
    "IT & Wi-Fi": [
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

    "Hostel": [
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

    "Academic": [
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

    "Infrastructure": [
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

    "Transport": [
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
    ]
}

places = [
    "Block A", "Block B", "Block C", "Block D",
    "Central Library", "Main Building", "Science Building",
    "Boys Hostel", "Girls Hostel", "Computer Lab",
    "Seminar Hall", "Workshop"
]

high_words = [
    "Urgent", "Emergency", "Fire", "Accident",
    "Unsafe", "Medical", "Security", "Immediately"
]

rows = []

# Generate 120 samples per category across 5 categories (600 rows total)
for category, category_templates in templates.items():

    for _ in range(120):

        sentence = random.choice(category_templates)
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
            "priority": priority
        })

random.shuffle(rows)

with open("data/complaints.csv", "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["complaint", "category", "priority"]
    )

    writer.writeheader()
    writer.writerows(rows)

print(f"Created {len(rows)} complaints.")
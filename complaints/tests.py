"""
Tests for the CAMPUSCARE staff dashboard (Issue #1).

Run with:
    python manage.py test complaints

Covers:
    - Student-facing complaint submission (unchanged behaviour)
    - Dashboard access control (unauthenticated, non-staff, staff)
    - Dashboard summary counts
    - Search and filter functionality
    - Status update endpoint
    - Complaint model defaults
"""

from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse

from .models import Complaint
from .ml import (
    predict_category,
    predict_category_with_confidence,
    predict_priority,
    detect_language,
    detect_duplicate,
    load_category_model,
    train_category_model,
    MODEL_PATH
)

User = get_user_model()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_complaint(**kwargs):
    """Create a Complaint with sensible defaults."""
    defaults = dict(
        student_name="Test Student",
        description="Wi-Fi is not working in Block A",
        category="IT & Wi-Fi",
        priority="Medium",
        department="IT Department",
        status="Open",
        is_duplicate=False,
    )
    defaults.update(kwargs)
    return Complaint.objects.create(**defaults)


# ---------------------------------------------------------------------------
# 1. Complaint model tests
# ---------------------------------------------------------------------------

class ComplaintModelTest(TestCase):

    def test_complaint_creation_stores_all_fields(self):
        """All Complaint fields are persisted correctly."""
        c = make_complaint()
        self.assertEqual(c.student_name, "Test Student")
        self.assertEqual(c.category, "IT & Wi-Fi")
        self.assertEqual(c.priority, "Medium")
        self.assertEqual(c.department, "IT Department")

    def test_default_status_is_open(self):
        """Newly created complaints default to Open status."""
        c = make_complaint()
        self.assertEqual(c.status, "Open")

    def test_default_is_not_duplicate(self):
        """Newly created complaints default to not-duplicate."""
        c = make_complaint()
        self.assertFalse(c.is_duplicate)

    def test_str_representation(self):
        """__str__ returns '<student_name> - <category>'."""
        c = make_complaint()
        self.assertEqual(str(c), "Test Student - IT & Wi-Fi")

    def test_status_can_be_updated(self):
        """Complaint status can be updated and persists to the DB."""
        c = make_complaint()
        c.status = "Resolved"
        c.save()
        refreshed = Complaint.objects.get(pk=c.pk)
        self.assertEqual(refreshed.status, "Resolved")
        
    def test_confidence_is_stored(self):
        """Confidence value is persisted correctly."""
        c = make_complaint(confidence=0.92)
        self.assertAlmostEqual(c.confidence, 0.92, places=2)
        
        
# ---------------------------------------------------------------------------
# 2. Student-facing home view (unchanged behaviour)
# ---------------------------------------------------------------------------

class HomeViewTest(TestCase):

    def setUp(self):
        self.client = Client()

    def test_get_home_returns_200(self):
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)

    def test_post_complaint_creates_record(self):
        self.client.post(reverse('home'), {
            'student_name': 'Alice',
            'description': 'Wi-Fi is not working in Block C',
        })
        self.assertEqual(Complaint.objects.count(), 1)

    def test_post_complaint_returns_200(self):
        response = self.client.post(reverse('home'), {
            'student_name': 'Alice',
            'description': 'Bus is arriving late',
        })
        self.assertEqual(response.status_code, 200)

    def test_post_complaint_result_in_context(self):
        response = self.client.post(reverse('home'), {
            'student_name': 'Alice',
            'description': 'Wi-Fi is not working in Block C',
        })
        self.assertIn('result', response.context)
        result = response.context['result']
        self.assertIn('category', result)
        self.assertIn('priority', result)
        self.assertIn('duplicate', result)

    def test_post_duplicate_complaint_flagged(self):
        payload = {
            'student_name': 'Alice',
            'description': 'Wi-Fi is not working in Block A hostel area',
        }
        self.client.post(reverse('home'), payload)
        response = self.client.post(reverse('home'), payload)
        self.assertTrue(response.context['result']['duplicate'])


# ---------------------------------------------------------------------------
# 3. Dashboard access control
# ---------------------------------------------------------------------------

class DashboardAccessControlTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.staff_user = User.objects.create_user(
            username='staffuser', password='testpass123', is_staff=True
        )
        self.regular_user = User.objects.create_user(
            username='student', password='testpass123', is_staff=False
        )

    def test_unauthenticated_redirected_to_login(self):
        """Unauthenticated request to /dashboard/ is redirected to login."""
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login', response['Location'])

    def test_non_staff_user_redirected(self):
        """A logged-in non-staff user cannot access the dashboard."""
        self.client.login(username='student', password='testpass123')
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 302)

    def test_staff_user_can_access_dashboard(self):
        """A staff user gets HTTP 200 on the dashboard."""
        self.client.login(username='staffuser', password='testpass123')
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)

    def test_login_page_accessible(self):
        """The login page is publicly accessible (HTTP 200)."""
        response = self.client.get(reverse('login'))
        self.assertEqual(response.status_code, 200)


# ---------------------------------------------------------------------------
# 4. Dashboard summary counts
# ---------------------------------------------------------------------------

class DashboardSummaryTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.staff = User.objects.create_user(
            username='staff', password='pass', is_staff=True
        )
        self.client.login(username='staff', password='pass')

    def _get_summary(self):
        response = self.client.get(reverse('dashboard'))
        return response.context['summary']

    def test_summary_total_count(self):
        """Summary total equals number of Complaint records."""
        make_complaint()
        make_complaint(description="Bus is late")
        summary = self._get_summary()
        self.assertEqual(summary['total'], 2)

    def test_summary_open_count(self):
        """Summary open count reflects Open complaints only."""
        make_complaint(status='Open')
        make_complaint(status='Resolved')
        summary = self._get_summary()
        self.assertEqual(summary['open'], 1)

    def test_summary_in_progress_count(self):
        """Summary in_progress count reflects In Progress complaints."""
        make_complaint(status='In Progress')
        make_complaint(status='Open')
        summary = self._get_summary()
        self.assertEqual(summary['in_progress'], 1)

    def test_summary_resolved_count(self):
        """Summary resolved count reflects Resolved complaints only."""
        make_complaint(status='Resolved')
        make_complaint(status='Open')
        summary = self._get_summary()
        self.assertEqual(summary['resolved'], 1)

    def test_summary_high_priority_count(self):
        """Summary high_priority count reflects High priority complaints."""
        make_complaint(priority='High')
        make_complaint(priority='Medium')
        make_complaint(priority='Low')
        summary = self._get_summary()
        self.assertEqual(summary['high_priority'], 1)

    def test_summary_duplicate_count(self):
        """Summary duplicates count reflects complaints flagged as duplicate."""
        make_complaint(is_duplicate=True)
        make_complaint(is_duplicate=False)
        summary = self._get_summary()
        self.assertEqual(summary['duplicates'], 1)

    def test_summary_zero_when_no_complaints(self):
        """All summary counts are 0 when there are no complaints."""
        summary = self._get_summary()
        self.assertEqual(summary['total'], 0)
        self.assertEqual(summary['high_priority'], 0)
        self.assertEqual(summary['duplicates'], 0)


# ---------------------------------------------------------------------------
# 5. Dashboard search and filter
# ---------------------------------------------------------------------------

class DashboardFilterTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.staff = User.objects.create_user(
            username='staff', password='pass', is_staff=True
        )
        self.client.login(username='staff', password='pass')

        # Seed mixed complaints
        make_complaint(description="Wi-Fi is down in the library", category="IT & Wi-Fi", priority="High", status="Open", department="IT Department")
        make_complaint(description="Bus is late every morning", category="Transport", priority="Medium", status="Open", department="Transport Office")
        make_complaint(description="Hostel room fan broken", category="Hostel", priority="Medium", status="Resolved", department="Hostel Management")
        make_complaint(description="Classroom projector not working", category="Infrastructure", priority="Low", status="In Progress", department="Maintenance")
        make_complaint(description="Exam marks not updated", category="Academic", priority="Low", status="Open", department="Academic Office")

    def _complaints(self, params):
        response = self.client.get(reverse('dashboard'), params)
        return list(response.context['complaints'])

    def test_no_filter_returns_all(self):
        """No filter returns all 5 complaints."""
        self.assertEqual(len(self._complaints({})), 5)

    def test_filter_by_category(self):
        """Filtering by category returns only matching complaints."""
        results = self._complaints({'category': 'Transport'})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].category, 'Transport')

    def test_filter_by_priority_high(self):
        """Filtering by priority=High returns only High complaints."""
        results = self._complaints({'priority': 'High'})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].priority, 'High')

    def test_filter_by_status_resolved(self):
        """Filtering by status=Resolved returns only Resolved complaints."""
        results = self._complaints({'status': 'Resolved'})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, 'Resolved')

    def test_filter_by_status_in_progress(self):
        """Filtering by status=In Progress returns only those complaints."""
        results = self._complaints({'status': 'In Progress'})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, 'In Progress')

    def test_filter_by_department(self):
        """Filtering by department returns only matching complaints."""
        results = self._complaints({'department': 'Maintenance'})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].department, 'Maintenance')

    def test_search_by_keyword(self):
        """Text search returns complaints whose description contains the keyword."""
        results = self._complaints({'q': 'library'})
        self.assertEqual(len(results), 1)
        self.assertIn('library', results[0].description.lower())

    def test_search_case_insensitive(self):
        """Text search is case-insensitive."""
        results = self._complaints({'q': 'BUS'})
        self.assertEqual(len(results), 1)
        self.assertIn('bus', results[0].description.lower())

    def test_search_no_match_returns_empty(self):
        """Text search with no match returns zero results."""
        results = self._complaints({'q': 'xyznosuchthing'})
        self.assertEqual(len(results), 0)

    def test_combined_filter_category_and_priority(self):
        """Combined category + priority filter works correctly."""
        results = self._complaints({'category': 'IT & Wi-Fi', 'priority': 'High'})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].category, 'IT & Wi-Fi')
        self.assertEqual(results[0].priority, 'High')

    def test_filter_no_match_returns_empty(self):
        """A filter that matches nothing returns an empty list."""
        results = self._complaints({'category': 'Academic', 'priority': 'High'})
        self.assertEqual(len(results), 0)


# ---------------------------------------------------------------------------
# 6. Status update endpoint
# ---------------------------------------------------------------------------

class StatusUpdateTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.staff = User.objects.create_user(
            username='staff', password='pass', is_staff=True
        )
        self.non_staff = User.objects.create_user(
            username='student', password='pass', is_staff=False
        )

    def test_status_update_persists(self):
        """POSTing a new status to /dashboard/update/<id>/ persists the change."""
        c = make_complaint(status='Open')
        self.client.login(username='staff', password='pass')
        self.client.post(
            reverse('update_status', args=[c.pk]),
            {'status': 'In Progress'},
        )
        c.refresh_from_db()
        self.assertEqual(c.status, 'In Progress')

    def test_status_update_to_resolved(self):
        """Status can be updated all the way to Resolved."""
        c = make_complaint(status='Open')
        self.client.login(username='staff', password='pass')
        self.client.post(
            reverse('update_status', args=[c.pk]),
            {'status': 'Resolved'},
        )
        c.refresh_from_db()
        self.assertEqual(c.status, 'Resolved')

    def test_status_update_redirects_to_dashboard(self):
        """After update, the response redirects to /dashboard/."""
        c = make_complaint()
        self.client.login(username='staff', password='pass')
        response = self.client.post(
            reverse('update_status', args=[c.pk]),
            {'status': 'In Progress'},
        )
        self.assertRedirects(response, reverse('dashboard'))

    def test_invalid_status_not_saved(self):
        """An invalid status value is rejected and the status stays unchanged."""
        c = make_complaint(status='Open')
        self.client.login(username='staff', password='pass')
        self.client.post(
            reverse('update_status', args=[c.pk]),
            {'status': 'InvalidStatus'},
        )
        c.refresh_from_db()
        self.assertEqual(c.status, 'Open')

    def test_unauthenticated_cannot_update_status(self):
        """Unauthenticated users cannot change complaint status."""
        c = make_complaint(status='Open')
        self.client.post(
            reverse('update_status', args=[c.pk]),
            {'status': 'Resolved'},
        )
        c.refresh_from_db()
        self.assertEqual(c.status, 'Open')

    def test_non_staff_cannot_update_status(self):
        """Non-staff users cannot change complaint status."""
        c = make_complaint(status='Open')
        self.client.login(username='student', password='pass')
        self.client.post(
            reverse('update_status', args=[c.pk]),
            {'status': 'Resolved'},
        )
        c.refresh_from_db()
        self.assertEqual(c.status, 'Open')

    def test_nonexistent_complaint_returns_404(self):
        """Update request for a non-existent complaint ID returns 404."""
        self.client.login(username='staff', password='pass')
        response = self.client.post(
            reverse('update_status', args=[99999]),
            {'status': 'Resolved'},
        )
        self.assertEqual(response.status_code, 404)


# ---------------------------------------------------------------------------
# 7. Dashboard context keys
# ---------------------------------------------------------------------------

class DashboardContextTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.staff = User.objects.create_user(
            username='staff', password='pass', is_staff=True
        )
        self.client.login(username='staff', password='pass')

    def test_context_contains_complaints(self):
        """Dashboard context includes 'complaints' queryset."""
        response = self.client.get(reverse('dashboard'))
        self.assertIn('complaints', response.context)

    def test_context_contains_summary(self):
        """Dashboard context includes 'summary' dict."""
        response = self.client.get(reverse('dashboard'))
        self.assertIn('summary', response.context)

    def test_context_contains_status_choices(self):
        """Dashboard context includes status_choices list."""
        response = self.client.get(reverse('dashboard'))
        self.assertIn('status_choices', response.context)
        self.assertIn('Open', response.context['status_choices'])
        self.assertIn('In Progress', response.context['status_choices'])
        self.assertIn('Resolved', response.context['status_choices'])

    def test_context_contains_categories(self):
        """Dashboard context includes all five complaint categories."""
        response = self.client.get(reverse('dashboard'))
        self.assertIn('categories', response.context)
        for cat in ['Academic', 'Hostel', 'Infrastructure', 'IT & Wi-Fi', 'Transport']:
            self.assertIn(cat, response.context['categories'])

    def test_complaints_ordered_newest_first(self):
        """Dashboard returns complaints ordered newest-first (by pk descending as proxy)."""
        c1 = make_complaint(description="First complaint")
        c2 = make_complaint(description="Second complaint")
        response = self.client.get(reverse('dashboard'))
        complaints = list(response.context['complaints'])
        pks = [c.pk for c in complaints]
        # c2 was inserted after c1, so it should appear before c1 in the list
        self.assertIn(c1.pk, pks)
        self.assertIn(c2.pk, pks)
        self.assertGreater(pks.index(c1.pk), pks.index(c2.pk))


# ---------------------------------------------------------------------------
# 8. ML prediction functions (unchanged baseline)
# ---------------------------------------------------------------------------

class MLPredictionTest(TestCase):

    def test_predict_category_it_wifi(self):
        self.assertEqual(predict_category("Wi-Fi is not working in Block C"), "IT & Wi-Fi")

    def test_predict_category_hostel(self):
        self.assertEqual(predict_category("There is no water supply in the hostel"), "Hostel")

    def test_predict_category_transport(self):
        self.assertEqual(predict_category("College bus is arriving late every morning"), "Transport")

    def test_predict_priority_high(self):
        self.assertEqual(predict_priority("Emergency! Fire near the chemistry lab"), "High")

    def test_predict_priority_medium(self):
        self.assertEqual(predict_priority("The projector is not working in Hall 3"), "Medium")

    def test_predict_priority_low(self):
        self.assertEqual(predict_priority("The canteen menu could use more variety"), "Low")

    def test_detect_duplicate_empty_db(self):
        self.assertFalse(detect_duplicate("Wi-Fi is down in Block A"))

    def test_detect_exact_duplicate(self):
        desc = "Wi-Fi is not working in Block A"
        make_complaint(description=desc)
        self.assertTrue(detect_duplicate(desc))

    def test_multilingual_complaint_creation(self):
        complaint = Complaint.objects.create(
            student_name="Rohit Sharma",
            description="हॉस्टल में पानी नहीं आ रहा है",
            category="Hostel",
            priority="High",
            department="Hostel Management",
            language="Hindi"
        )
        self.assertEqual(complaint.language, "Hindi")
        self.assertEqual(complaint.category, "Hostel")

    def test_complaint_default_confidence(self):
        complaint = Complaint.objects.create(
            student_name="Jane Doe",
            description="Hostel room tap is leaking"
        )
        self.assertEqual(complaint.confidence, 1.0)


class ComplaintLanguageDetectionTest(TestCase):

    def test_english_detection(self):
        self.assertEqual(
            detect_language("Wi-Fi is not working in Block C"),
            "English"
        )
        self.assertEqual(
            detect_language("Emergency! There is no water supply in hostel"),
            "English"
        )

    def test_hindi_detection(self):
        self.assertEqual(
            detect_language("हॉस्टल में पानी नहीं आ रहा है"),
            "Hindi"
        )
        self.assertEqual(
            detect_language("क्लासरूम का प्रोजेक्टर काम नहीं कर रहा है"),
            "Hindi"
        )

    def test_hinglish_detection(self):
        self.assertEqual(
            detect_language("Hostel mein paani nahi aa raha hai"),
            "Hinglish"
        )
        self.assertEqual(
            detect_language("Block C mein wifi kaam nahi kar raha"),
            "Hinglish"
        )

    def test_marathi_detection(self):
        self.assertEqual(
            detect_language("वसतिगृहात पाणी येत नाही आहे"),
            "Marathi"
        )
        self.assertEqual(
            detect_language("वर्गातील प्रोजेक्टर व्यवस्थित चालत नाही आहे"),
            "Marathi"
        )


class ComplaintMLCategoryTest(TestCase):

    # English Category Tests
    def test_english_it_prediction(self):
        self.assertEqual(
            predict_category("Wi-Fi is not working in Block C"),
            "IT & Wi-Fi"
        )

    def test_english_hostel_prediction(self):
        self.assertEqual(
            predict_category("There is no water supply in the hostel"),
            "Hostel"
        )

    def test_english_infrastructure_prediction(self):
        self.assertEqual(
            predict_category("Classroom projector is damaged in Main Building"),
            "Infrastructure"
        )

    def test_english_transport_prediction(self):
        self.assertEqual(
            predict_category("College bus is arriving late"),
            "Transport"
        )

    def test_english_academic_prediction(self):
        self.assertEqual(
            predict_category("Internal marks are missing on portal"),
            "Academic"
        )

    def test_it_wifi_prediction(self):
        category = predict_category("Wi-Fi is not working in Block C")
        self.assertEqual(category, "IT & Wi-Fi")

    def test_hostel_prediction(self):
        category = predict_category("There is no water supply in the hostel")
        self.assertEqual(category, "Hostel")

    def test_academic_prediction(self):
        category = predict_category("Internal marks are missing on student dashboard")
        self.assertEqual(category, "Academic")

    def test_infrastructure_prediction(self):
        category = predict_category("Classroom projector bulb is fused in Main Building")
        self.assertEqual(category, "Infrastructure")

    def test_transport_prediction(self):
        category = predict_category("College bus is arriving late every morning")
        self.assertEqual(category, "Transport")

    def test_eduroam_laptop_prediction(self):
        category = predict_category("Cannot connect to eduroam on my laptop in central library")
        self.assertEqual(category, "IT & Wi-Fi")

    def test_geyser_laundry_prediction(self):
        category = predict_category("Hostel geyser not heating water in 2nd floor bathroom")
        self.assertEqual(category, "Hostel")

    def test_air_conditioner_leak_prediction(self):
        category = predict_category("Air conditioner in seminar hall is leaking water and malfunctioning")
        self.assertEqual(category, "Infrastructure")

    def test_lecture_attendance_prediction(self):
        category = predict_category("Professor cancelled lecture without notice and attendance marked absent")
        self.assertEqual(category, "Academic")

    def test_shuttle_van_delay_prediction(self):
        category = predict_category("College shuttle van from metro station delayed by 45 minutes")
        self.assertEqual(category, "Transport")

    # Hindi Category Tests
    def test_hindi_it_prediction(self):
        self.assertEqual(
            predict_category("ब्लॉक सी में वाई-फाई काम नहीं कर रहा है"),
            "IT & Wi-Fi"
        )

    def test_hindi_hostel_prediction(self):
        self.assertEqual(
            predict_category("हॉस्टल में पानी की कोई सप्लाई नहीं आ रही है"),
            "Hostel"
        )

    def test_hindi_infrastructure_prediction(self):
        self.assertEqual(
            predict_category("क्लासरूम का प्रोजेक्टर काम नहीं कर रहा है"),
            "Infrastructure"
        )

    def test_hindi_transport_prediction(self):
        self.assertEqual(
            predict_category("कॉलेज बस बहुत देरी से आ रही है"),
            "Transport"
        )

    def test_hindi_academic_prediction(self):
        self.assertEqual(
            predict_category("पोर्टल पर मेरे मार्क्स नहीं दिख रहे हैं"),
            "Academic"
        )

    # Hinglish Category Tests
    def test_hinglish_it_prediction(self):
        self.assertEqual(
            predict_category("Block C mein wifi kaam nahi kar raha hai"),
            "IT & Wi-Fi"
        )

    def test_hinglish_hostel_prediction(self):
        self.assertEqual(
            predict_category("Hostel mein paani bilkul nahi aa raha hai"),
            "Hostel"
        )

    def test_hinglish_infrastructure_prediction(self):
        self.assertEqual(
            predict_category("Classroom projector kaam nahi kar raha hai"),
            "Infrastructure"
        )

    def test_hinglish_transport_prediction(self):
        self.assertEqual(
            predict_category("College bus bohot late aa rahi hai"),
            "Transport"
        )

    def test_hinglish_academic_prediction(self):
        self.assertEqual(
            predict_category("Portal par mere marks show nahi ho rahe hain"),
            "Academic"
        )

    # Marathi Category Tests
    def test_marathi_it_prediction(self):
        self.assertEqual(
            predict_category("ब्लॉक सी मध्ये वाय-फाय व्यवस्थित चालत नाही आहे"),
            "IT & Wi-Fi"
        )

    def test_marathi_hostel_prediction(self):
        self.assertEqual(
            predict_category("वसतिगृहात पाण्याचा पुरवठा बंद आहे"),
            "Hostel"
        )

    def test_marathi_infrastructure_prediction(self):
        self.assertEqual(
            predict_category("वर्गातील प्रोजेक्टर व्यवस्थित चालत नाही"),
            "Infrastructure"
        )

    def test_marathi_transport_prediction(self):
        self.assertEqual(
            predict_category("महाविद्यालयाची बस खूप उशिरा येत आहे"),
            "Transport"
        )

    def test_marathi_academic_prediction(self):
        self.assertEqual(
            predict_category("पोर्टलवर परीक्षेचे गुण दिसत नाहीत"),
            "Academic"
        )


class ComplaintMLConfidenceTest(TestCase):

    def test_confidence_return_structure(self):
        category, confidence = predict_category_with_confidence("Wi-Fi is down in Block C")
        self.assertEqual(category, "IT & Wi-Fi")
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_high_confidence_on_clear_samples(self):
        _, confidence = predict_category_with_confidence("College bus is arriving late every day")
        self.assertGreater(confidence, 0.50)


class ComplaintMLPriorityTest(TestCase):

    # High Priority Cases
    def test_english_priority(self):
        self.assertEqual(
            predict_priority("Emergency! There is no water supply"),
            "High"
        )
        self.assertEqual(
            predict_priority("General query about library hours"),
            "Low"
        )

    def test_electric_shock_high_priority(self):
        self.assertEqual(
            predict_priority("Electric shock felt from metal switchboard in physics laboratory"),
            "High"
        )

    def test_food_poisoning_high_priority(self):
        self.assertEqual(
            predict_priority("Hostel mess food quality is spoiled and caused students vomiting"),
            "High"
        )

    def test_ceiling_collapse_high_priority(self):
        self.assertEqual(
            predict_priority("Ceiling plaster fell down on student desk in classroom 101"),
            "High"
        )

    def test_hall_ticket_deadline_high_priority(self):
        self.assertEqual(
            predict_priority("Hall ticket for upcoming end-semester exam is not generated"),
            "High"
        )

    def test_hindi_priority(self):
        self.assertEqual(
            predict_priority("आपातकालीन: हॉस्टल में पानी नहीं आ रहा"),
            "High"
        )
        self.assertEqual(
            predict_priority("क्लासरूम का पंखा काम नहीं कर रहा"),
            "Medium"
        )
        self.assertEqual(
            predict_priority("लाइब्रेरी के बारे में जानकारी चाहिए"),
            "Low"
        )

    def test_hinglish_priority(self):
        self.assertEqual(
            predict_priority("Khatra: hostel mein open wire hai"),
            "High"
        )
        self.assertEqual(
            predict_priority("Classroom ka projector kharab hai"),
            "Medium"
        )
        self.assertEqual(
            predict_priority("Timetable ki request"),
            "Low"
        )

    def test_marathi_priority(self):
        self.assertEqual(
            predict_priority("आणीबाणी: वसतिगृहात पाणी नाही"),
            "High"
        )
        self.assertEqual(
            predict_priority("वर्गातील पंखा बंद पडले आहे"),
            "Medium"
        )
        self.assertEqual(
            predict_priority("ग्रंथालयाची माहिती हवी आहे"),
            "Low"
        )

    # Medium Priority Cases
    def test_wifi_not_working_medium_priority(self):
        self.assertEqual(
            predict_priority("Wi-Fi is not working in Block A"),
            "Medium"
        )

    def test_projector_fused_medium_priority(self):
        self.assertEqual(
            predict_priority("Classroom projector bulb is fused in main block"),
            "Medium"
        )

    def test_tap_dripping_medium_priority(self):
        self.assertEqual(
            predict_priority("Bathroom tap is continuously dripping and wasting water"),
            "Medium"
        )

    def test_course_slides_missing_medium_priority(self):
        self.assertEqual(
            predict_priority("Course syllabus and lecture slides not uploaded on learning portal"),
            "Medium"
        )

    # Low Priority & Negation Cases
    def test_general_inquiry_low_priority(self):
        self.assertEqual(
            predict_priority("Inquiry regarding bus routes for the upcoming semester"),
            "Low"
        )

    def test_negation_not_an_emergency_low_priority(self):
        self.assertEqual(
            predict_priority("This is not an emergency, just need information on bus fee payment"),
            "Low"
        )


class ComplaintDuplicateDetectionTest(TestCase):

    def test_duplicate_detection(self):
        Complaint.objects.create(
            student_name="Student 1",
            description="Wi-Fi is not working in Block C",
            category="IT & Wi-Fi",
            priority="Medium",
            department="IT Department"
        )

        # Very similar complaint should be flagged
        is_dup = detect_duplicate("Wi-Fi is not working in Block C")
        self.assertTrue(is_dup)

        # Dissimilar complaint should not be flagged
        is_not_dup = detect_duplicate("Hostel food quality is terrible")
        self.assertFalse(is_not_dup)

    def test_duplicate_detection_multilingual(self):
        Complaint.objects.create(
            student_name="Student 1",
            description="हॉस्टल में पानी नहीं आ रहा है",
            category="Hostel",
            priority="High",
            department="Hostel Management",
            language="Hindi"
        )

        # Similar complaint in Hindi should be flagged
        is_dup = detect_duplicate("हॉस्टल में पानी नहीं आ रहा")
        self.assertTrue(is_dup)

        # Unrelated complaint should not be flagged
        is_not_dup = detect_duplicate("College bus is late")
        self.assertFalse(is_not_dup)


class ComplaintViewTest(TestCase):

    def setUp(self):
        self.client = Client()

    def test_home_page_get(self):
        response = self.client.get(reverse("home"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "CampusCare")

    def test_home_page_post_creates_complaint_with_confidence(self):
        response = self.client.post(
            reverse("home"),
            {
                "student_name": "Rohan Gupta",
                "description": "College bus is arriving late every morning"
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Transport")
        self.assertContains(response, "Confidence")

        complaint = Complaint.objects.last()
        self.assertEqual(complaint.student_name, "Rohan Gupta")
        self.assertEqual(complaint.category, "Transport")
        self.assertEqual(complaint.department, "Transport Office")
        self.assertGreater(complaint.confidence, 0.0)

    def test_home_page_post_hindi(self):
        response = self.client.post(
            reverse("home"),
            {
                "student_name": "Aarav Sharma",
                "description": "हॉस्टल में पानी की कोई सप्लाई नहीं आ रही है"
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Hostel")
        self.assertContains(response, "Hindi")

        complaint = Complaint.objects.last()
        self.assertEqual(complaint.student_name, "Aarav Sharma")
        self.assertEqual(complaint.category, "Hostel")
        self.assertEqual(complaint.department, "Hostel Management")
        self.assertEqual(complaint.language, "Hindi")

    def test_home_page_post_hinglish(self):
        response = self.client.post(
            reverse("home"),
            {
                "student_name": "Priya Verma",
                "description": "Block C mein wifi kaam nahi kar raha hai"
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "IT &amp; Wi-Fi")
        self.assertContains(response, "Hinglish")

        complaint = Complaint.objects.last()
        self.assertEqual(complaint.language, "Hinglish")
        self.assertEqual(complaint.category, "IT & Wi-Fi")
        self.assertEqual(complaint.department, "IT Department")

    def test_home_page_post_marathi(self):
        response = self.client.post(
            reverse("home"),
            {
                "student_name": "Tanmay Patil",
                "description": "वसतिगृहात पाण्याचा पुरवठा बंद आहे"
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Hostel")
        self.assertContains(response, "Marathi")

        complaint = Complaint.objects.last()
        self.assertEqual(complaint.language, "Marathi")
        self.assertEqual(complaint.category, "Hostel")


class ComplaintDuplicateDetectionTest(TestCase):

    def setUp(self):
        Complaint.objects.create(
            student_name="Alice",
            description="Wi-Fi is not working in Block C",
            category="IT & Wi-Fi",
            department="IT Department"
        )
        Complaint.objects.create(
            student_name="Bob",
            description="There is no water supply in hostel block B",
            category="Hostel",
            department="Hostel Management"
        )

    def test_empty_input_returns_false(self):
        self.assertFalse(detect_duplicate(""))
        self.assertFalse(detect_duplicate("   "))
        self.assertFalse(detect_duplicate(None))

    def test_exact_duplicate_detection(self):
        self.assertTrue(detect_duplicate("Wi-Fi is not working in Block C"))
        self.assertTrue(detect_duplicate("There is no water supply in hostel block B"))

    def test_rephrased_duplicate_detection(self):
        self.assertTrue(detect_duplicate("No water supply in hostel block B"))

    def test_distinct_complaint_returns_false(self):
        self.assertFalse(detect_duplicate("Projector bulb is fused in classroom 204"))

    def test_category_scoped_duplicate_detection(self):
        self.assertTrue(detect_duplicate("Wi-Fi is not working in Block C", category="IT & Wi-Fi"))
        self.assertFalse(detect_duplicate("Wi-Fi is not working in Block C", category="Hostel"))


class ComplaintModelPersistenceTest(TestCase):

    def test_model_artifact_exists_and_loads(self):
        import os
        self.assertTrue(os.path.exists(MODEL_PATH), "category_model.joblib should exist in data/")
        vec, model = load_category_model(model_path=MODEL_PATH, fallback_train=False)
        self.assertIsNotNone(vec)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(vec, "transform"))

    def test_prediction_using_persisted_model(self):
        category = predict_category("Wi-Fi is not working in central library")
        self.assertEqual(category, "IT & Wi-Fi")


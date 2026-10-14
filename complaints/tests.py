"""
Complaint evaluation tests for CAMPUSCARE.

Run with:
    python manage.py test complaints

These tests verify that category and priority predictions behave as expected
for representative examples across all supported categories and priority levels.

NOTE: This hand-written evaluation set checks known representative inputs only.
It does not prove real-world model accuracy on unseen data.
"""

from django.test import Client, TestCase

from .ml import detect_duplicate, predict_category, predict_priority
from .models import Complaint


# ---------------------------------------------------------------------------
# 1. Model field tests
# ---------------------------------------------------------------------------

class ComplaintModelTest(TestCase):
    """Basic sanity checks for the Complaint database model."""

    def test_complaint_creation_stores_all_fields(self):
        """Complaint is persisted with every field correctly."""
        complaint = Complaint.objects.create(
            student_name="Test Student",
            description="Wi-Fi is not working in Block A",
            category="IT & Wi-Fi",
            priority="Medium",
            department="IT Department",
        )
        self.assertEqual(complaint.student_name, "Test Student")
        self.assertEqual(complaint.description, "Wi-Fi is not working in Block A")
        self.assertEqual(complaint.category, "IT & Wi-Fi")
        self.assertEqual(complaint.priority, "Medium")
        self.assertEqual(complaint.department, "IT Department")

    def test_complaint_default_status_is_open(self):
        """Newly created complaints default to Open status."""
        complaint = Complaint.objects.create(
            student_name="Alice",
            description="Canteen food quality is poor",
            category="Hostel",
            priority="Low",
            department="Hostel Office",
        )
        self.assertEqual(complaint.status, "Open")

    def test_complaint_default_is_not_duplicate(self):
        """Newly created complaints default to not-duplicate."""
        complaint = Complaint.objects.create(
            student_name="Bob",
            description="Bus is late every day",
            category="Transport",
            priority="Medium",
            department="Transport Office",
        )
        self.assertFalse(complaint.is_duplicate)

    def test_complaint_str_representation(self):
        """__str__ returns '<student_name> - <category>'."""
        complaint = Complaint.objects.create(
            student_name="Carol",
            description="Projector not working",
            category="IT & Wi-Fi",
            priority="Medium",
            department="IT Department",
        )
        self.assertEqual(str(complaint), "Carol - IT & Wi-Fi")

    def test_multiple_complaints_stored_independently(self):
        """Two different complaints are stored as separate records."""
        Complaint.objects.create(
            student_name="Dave",
            description="Library is closed during exams",
            category="Academic",
            priority="High",
            department="Academic Office",
        )
        Complaint.objects.create(
            student_name="Eve",
            description="Ceiling fan broken in Room 301",
            category="Hostel",
            priority="Medium",
            department="Hostel Office",
        )
        self.assertEqual(Complaint.objects.count(), 2)


# ---------------------------------------------------------------------------
# 2. Category prediction tests — one clear example per supported category
# ---------------------------------------------------------------------------

class CategoryPredictionTest(TestCase):
    """
    Representative examples covering every supported complaint category.

    Supported categories (from the training dataset):
        Academic, Hostel, Infrastructure, IT & Wi-Fi, Transport
    """

    # -- Academic --

    def test_category_academic_marks(self):
        """Grade / marks complaint is classified as Academic."""
        self.assertEqual(
            predict_category("My internal marks have not been updated on the portal"),
            "Academic",
        )

    def test_category_academic_exam_schedule(self):
        """Exam timetable complaint is classified as Academic."""
        self.assertEqual(
            predict_category("The exam schedule conflicts with another paper"),
            "Academic",
        )

    def test_category_academic_course_registration(self):
        """Course registration issue is classified as Academic."""
        self.assertEqual(
            predict_category("I am unable to register for the elective course"),
            "Academic",
        )

    # -- Hostel --

    def test_category_hostel_water_supply(self):
        """Water supply issue in hostel is classified as Hostel."""
        self.assertEqual(
            predict_category("There is no water supply in the hostel block"),
            "Hostel",
        )

    def test_category_hostel_room_maintenance(self):
        """Hostel room maintenance complaint is classified as Hostel."""
        self.assertEqual(
            predict_category("The ceiling fan in my hostel room is broken"),
            "Hostel",
        )

    def test_category_hostel_warden(self):
        """Hostel warden-related complaint is classified as Hostel."""
        self.assertEqual(
            predict_category("Hostel warden is not addressing our maintenance requests"),
            "Hostel",
        )

    # -- Infrastructure --

    def test_category_infrastructure_classroom(self):
        """Classroom facility complaint is classified as Infrastructure."""
        self.assertEqual(
            predict_category("The classroom projector is not working"),
            "Infrastructure",
        )

    def test_category_infrastructure_washroom(self):
        """Washroom / sanitation facility complaint is classified as Infrastructure."""
        self.assertEqual(
            predict_category("The college building washroom has broken taps and damaged tiles"),
            "Infrastructure",
        )

    def test_category_infrastructure_door(self):
        """Broken classroom door is classified as Infrastructure."""
        self.assertEqual(
            predict_category("The classroom door is broken and cannot be locked"),
            "Infrastructure",
        )

    # -- IT & Wi-Fi --

    def test_category_it_wifi_network(self):
        """Wi-Fi network complaint is classified as IT & Wi-Fi."""
        self.assertEqual(
            predict_category("Wi-Fi is not working in Block C"),
            "IT & Wi-Fi",
        )

    def test_category_it_wifi_portal(self):
        """Wi-Fi login portal issue is classified as IT & Wi-Fi."""
        self.assertEqual(
            predict_category("The campus Wi-Fi login portal is down and internet is inaccessible"),
            "IT & Wi-Fi",
        )

    def test_category_it_wifi_slow_internet(self):
        """Slow internet complaint is classified as IT & Wi-Fi."""
        self.assertEqual(
            predict_category("The internet speed is very slow in the library"),
            "IT & Wi-Fi",
        )

    # -- Transport --

    def test_category_transport_bus_late(self):
        """Late bus complaint is classified as Transport."""
        self.assertEqual(
            predict_category("College bus is arriving late every morning"),
            "Transport",
        )

    def test_category_transport_route(self):
        """Bus route complaint is classified as Transport."""
        self.assertEqual(
            predict_category("The bus is skipping the Block A stop"),
            "Transport",
        )

    def test_category_transport_driver(self):
        """Bus driver complaint is classified as Transport."""
        self.assertEqual(
            predict_category("The bus driver is reckless and drives too fast"),
            "Transport",
        )


# ---------------------------------------------------------------------------
# 3. Priority prediction tests — High / Medium / Low
# ---------------------------------------------------------------------------

class PriorityPredictionTest(TestCase):
    """
    Representative examples for each priority level.

    Priority is keyword-based; these tests document the current behaviour
    and will fail visibly if the keyword list changes unexpectedly.
    """

    # -- High priority --

    def test_priority_high_emergency(self):
        """'emergency' keyword triggers High priority."""
        self.assertEqual(
            predict_priority("This is an emergency — student collapsed in the corridor"),
            "High",
        )

    def test_priority_high_fire(self):
        """'fire' keyword triggers High priority."""
        self.assertEqual(
            predict_priority("There is a fire near the chemistry lab"),
            "High",
        )

    def test_priority_high_urgent(self):
        """'urgent' keyword triggers High priority."""
        self.assertEqual(
            predict_priority("Urgent: water pipe has burst in the hostel"),
            "High",
        )

    def test_priority_high_danger(self):
        """'danger' keyword triggers High priority."""
        self.assertEqual(
            predict_priority("Exposed wiring in Room 201 is a danger to students"),
            "High",
        )

    def test_priority_high_medical(self):
        """'medical' keyword triggers High priority."""
        self.assertEqual(
            predict_priority("Medical attention needed for student in Block D"),
            "High",
        )

    def test_priority_high_no_water(self):
        """'no water' phrase triggers High priority."""
        self.assertEqual(
            predict_priority("There is no water in the entire hostel wing"),
            "High",
        )

    def test_priority_high_power_outage(self):
        """'power outage' phrase triggers High priority."""
        self.assertEqual(
            predict_priority("Power outage in Block B since last night"),
            "High",
        )

    # -- Medium priority --

    def test_priority_medium_not_working(self):
        """'not working' phrase triggers Medium priority."""
        self.assertEqual(
            predict_priority("The projector in Hall 3 is not working"),
            "Medium",
        )

    def test_priority_medium_broken(self):
        """'broken' keyword triggers Medium priority."""
        self.assertEqual(
            predict_priority("The bench in the library is broken"),
            "Medium",
        )

    def test_priority_medium_slow(self):
        """'slow' keyword triggers Medium priority."""
        self.assertEqual(
            predict_priority("Internet is slow in the computer lab"),
            "Medium",
        )

    def test_priority_medium_problem(self):
        """'problem' keyword triggers Medium priority."""
        self.assertEqual(
            predict_priority("There is a problem with my attendance record"),
            "Medium",
        )

    def test_priority_medium_issue(self):
        """'issue' keyword triggers Medium priority."""
        self.assertEqual(
            predict_priority("I am facing an issue with the fee payment gateway"),
            "Medium",
        )

    def test_priority_medium_damaged(self):
        """'damaged' keyword triggers Medium priority."""
        self.assertEqual(
            predict_priority("The lab equipment is damaged and needs repair"),
            "Medium",
        )

    def test_priority_medium_missing(self):
        """'missing' keyword triggers Medium priority."""
        self.assertEqual(
            predict_priority("My attendance marks are missing from the portal"),
            "Medium",
        )

    # -- Low priority --

    def test_priority_low_general_feedback(self):
        """General feedback with no trigger words defaults to Low."""
        self.assertEqual(
            predict_priority("The canteen menu could use more variety"),
            "Low",
        )

    def test_priority_low_suggestion(self):
        """A suggestion with no urgency words defaults to Low."""
        self.assertEqual(
            predict_priority("It would be nice to have more seating in the library"),
            "Low",
        )

    def test_priority_low_inquiry(self):
        """A neutral inquiry defaults to Low."""
        self.assertEqual(
            predict_priority("I would like to know the bus timetable for next week"),
            "Low",
        )


# ---------------------------------------------------------------------------
# 4. Edge case tests
# ---------------------------------------------------------------------------

class EdgeCaseTest(TestCase):
    """
    Edge cases for robustness: empty input, whitespace-only, very long text,
    numeric-only, and special characters.  These tests document current
    behaviour without asserting a specific category — they simply verify that
    the functions return a non-empty string and do not raise an exception.

    NOTE: The model returns one of the known categories even for nonsensical
    input because it always picks the most probable class.
    """

    VALID_CATEGORIES = {"Academic", "Hostel", "Infrastructure", "IT & Wi-Fi", "Transport"}
    VALID_PRIORITIES = {"High", "Medium", "Low"}

    def _assert_valid_category(self, text, label=""):
        """Helper: predict_category must return a known category without raising."""
        result = predict_category(text)
        self.assertIn(
            result,
            self.VALID_CATEGORIES,
            msg=f"predict_category({label!r}) returned unexpected value {result!r}",
        )

    def _assert_valid_priority(self, text, label=""):
        """Helper: predict_priority must return a known priority without raising."""
        result = predict_priority(text)
        self.assertIn(
            result,
            self.VALID_PRIORITIES,
            msg=f"predict_priority({label!r}) returned unexpected value {result!r}",
        )

    def test_edge_empty_string_category(self):
        """predict_category handles an empty string without raising."""
        self._assert_valid_category("", label="empty string")

    def test_edge_whitespace_only_category(self):
        """predict_category handles whitespace-only input without raising."""
        self._assert_valid_category("   ", label="whitespace only")

    def test_edge_whitespace_only_priority(self):
        """predict_priority handles whitespace-only input without raising."""
        self._assert_valid_priority("   ", label="whitespace only")

    def test_edge_very_long_text_category(self):
        """predict_category handles a 500-word complaint without raising."""
        long_text = ("The Wi-Fi connection is very slow and unreliable. " * 50).strip()
        self._assert_valid_category(long_text, label="500-word text")

    def test_edge_very_long_text_priority(self):
        """predict_priority handles a 500-word complaint without raising."""
        long_text = ("The Wi-Fi connection is very slow and unreliable. " * 50).strip()
        self._assert_valid_priority(long_text, label="500-word text")

    def test_edge_special_characters_category(self):
        """predict_category handles special characters without raising."""
        self._assert_valid_category(
            "Wi-Fi!!! @#$% not working???", label="special chars"
        )

    def test_edge_special_characters_priority(self):
        """predict_priority handles special characters without raising."""
        self._assert_valid_priority(
            "Emergency!!! @#$%^&*()", label="special chars"
        )

    def test_edge_numeric_only_category(self):
        """predict_category handles a numeric string without raising."""
        self._assert_valid_category("1234567890", label="numeric only")

    def test_edge_numeric_only_priority(self):
        """predict_priority handles a numeric string without raising."""
        self._assert_valid_priority("1234567890", label="numeric only")

    def test_edge_mixed_case_priority_high(self):
        """predict_priority is case-insensitive for trigger words."""
        self.assertEqual(
            predict_priority("EMERGENCY in Block A right now!"),
            "High",
        )

    def test_edge_mixed_case_priority_medium(self):
        """predict_priority is case-insensitive for medium-level words."""
        self.assertEqual(
            predict_priority("There is a PROBLEM with the hostel gate"),
            "Medium",
        )

    def test_edge_single_word_category(self):
        """predict_category handles a single keyword without raising."""
        self._assert_valid_category("hostel", label="single word")

    def test_edge_single_word_priority(self):
        """predict_priority handles a single keyword without raising."""
        self._assert_valid_priority("emergency", label="single word")


# ---------------------------------------------------------------------------
# 5. Duplicate detection tests
# ---------------------------------------------------------------------------

class DuplicateDetectionTest(TestCase):
    """Tests for the cosine-similarity-based duplicate detector."""

    def test_no_duplicate_when_db_empty(self):
        """Returns False when there are no existing complaints."""
        self.assertFalse(
            detect_duplicate("Wi-Fi is down in Block A")
        )

    def test_exact_duplicate_detected(self):
        """An exact re-submission is flagged as a duplicate."""
        description = "Wi-Fi is not working in Block A"
        Complaint.objects.create(
            student_name="Alice",
            description=description,
            category="IT & Wi-Fi",
            priority="Medium",
            department="IT Department",
        )
        self.assertTrue(detect_duplicate(description))

    def test_near_duplicate_detected(self):
        """A paraphrase of an existing complaint is flagged as a duplicate."""
        Complaint.objects.create(
            student_name="Bob",
            description="Wi-Fi is not working in Block A",
            category="IT & Wi-Fi",
            priority="Medium",
            department="IT Department",
        )
        self.assertTrue(
            detect_duplicate("Wi-Fi isn't working in Block A")
        )

    def test_different_complaint_not_duplicate(self):
        """A clearly different complaint is not flagged as a duplicate."""
        Complaint.objects.create(
            student_name="Carol",
            description="Bus is late every morning",
            category="Transport",
            priority="Medium",
            department="Transport Office",
        )
        self.assertFalse(
            detect_duplicate("The classroom projector bulb is broken")
        )

    def test_multiple_existing_not_duplicate(self):
        """A unique complaint is not duplicate even when many others exist."""
        for i in range(5):
            Complaint.objects.create(
                student_name=f"Student {i}",
                description=f"Bus stop {i} is not clean",
                category="Transport",
                priority="Low",
                department="Transport Office",
            )
        self.assertFalse(
            detect_duplicate("My hostel room has a water leakage problem")
        )


# ---------------------------------------------------------------------------
# 6. Prediction consistency tests
# ---------------------------------------------------------------------------

class PredictionConsistencyTest(TestCase):
    """
    Verifies that the ML model is deterministic: calling predict_category
    or predict_priority multiple times with the same input always returns
    the same output.
    """

    def test_category_prediction_is_deterministic(self):
        """predict_category returns the same result on repeated calls."""
        text = "The Wi-Fi in the library is not working"
        results = {predict_category(text) for _ in range(5)}
        self.assertEqual(
            len(results),
            1,
            msg=f"predict_category returned different results across calls: {results}",
        )

    def test_priority_prediction_is_deterministic(self):
        """predict_priority returns the same result on repeated calls."""
        text = "Emergency: student injured near the main gate"
        results = {predict_priority(text) for _ in range(5)}
        self.assertEqual(
            len(results),
            1,
            msg=f"predict_priority returned different results across calls: {results}",
        )

    def test_category_return_type_is_string(self):
        """predict_category always returns a str, never None."""
        result = predict_category("Bus is late today")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_priority_return_type_is_string(self):
        """predict_priority always returns a str, never None."""
        result = predict_priority("The door is broken")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


# ---------------------------------------------------------------------------
# 7. End-to-end view tests
# ---------------------------------------------------------------------------

class HomeViewTest(TestCase):
    """Integration tests for the complaint submission view (GET and POST)."""

    def setUp(self):
        self.client = Client()

    def test_get_home_returns_200(self):
        """GET / returns HTTP 200."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_post_valid_complaint_creates_record(self):
        """POSTing a valid complaint creates exactly one DB record."""
        self.client.post("/", {
            "student_name": "Test Student",
            "description": "Wi-Fi is not working in Block C",
        })
        self.assertEqual(Complaint.objects.count(), 1)

    def test_post_complaint_returns_200(self):
        """POSTing a complaint returns HTTP 200."""
        response = self.client.post("/", {
            "student_name": "Test Student",
            "description": "The bus is always late",
        })
        self.assertEqual(response.status_code, 200)

    def test_post_complaint_category_in_context(self):
        """Response context contains 'result' with category and priority."""
        response = self.client.post("/", {
            "student_name": "Test Student",
            "description": "Wi-Fi is not working in Block C",
        })
        self.assertIn("result", response.context)
        result = response.context["result"]
        self.assertIn("category", result)
        self.assertIn("priority", result)

    def test_post_complaint_is_duplicate_flag_in_context(self):
        """Response context result includes duplicate flag."""
        response = self.client.post("/", {
            "student_name": "Test Student",
            "description": "Bus is arriving late every day",
        })
        result = response.context["result"]
        self.assertIn("duplicate", result)

    def test_post_second_identical_complaint_flagged_as_duplicate(self):
        """Submitting the same complaint twice flags the second as duplicate."""
        payload = {
            "student_name": "Test Student",
            "description": "Wi-Fi is not working in Block A hostel",
        }
        self.client.post("/", payload)
        response = self.client.post("/", payload)
        result = response.context["result"]
        self.assertTrue(result["duplicate"])

    def test_post_academic_complaint_classified_correctly(self):
        """End-to-end: academic complaint is classified as Academic."""
        response = self.client.post("/", {
            "student_name": "Student A",
            "description": "My grade has not been updated after revaluation",
        })
        result = response.context["result"]
        self.assertEqual(result["category"], "Academic")

    def test_post_transport_complaint_classified_correctly(self):
        """End-to-end: transport complaint is classified as Transport."""
        response = self.client.post("/", {
            "student_name": "Student B",
            "description": "The college bus skipped my stop today",
        })
        result = response.context["result"]
        self.assertEqual(result["category"], "Transport")

    def test_post_high_priority_complaint_in_context(self):
        """End-to-end: urgent complaint is tagged as High priority."""
        response = self.client.post("/", {
            "student_name": "Student C",
            "description": "Emergency: gas leak detected near the chemistry lab",
        })
        result = response.context["result"]
        self.assertEqual(result["priority"], "High")
from django.contrib.auth.decorators import login_required, user_passes_test
from django.shortcuts import get_object_or_404, redirect, render

from .ml import (
    detect_duplicate,
    predict_category,
    predict_category_with_confidence,
    predict_priority,
    detect_language
)
from .models import Complaint


# ---------------------------------------------------------------------------
# Helper: restrict dashboard to staff users only
# ---------------------------------------------------------------------------

def _is_staff(user):
    return user.is_active and user.is_staff


staff_required = user_passes_test(_is_staff, login_url='login')


# ---------------------------------------------------------------------------
# Student-facing complaint submission view
# ---------------------------------------------------------------------------


def home(request):
    result = None

    if request.method == "POST":
        student_name = request.POST.get("student_name")
        description = request.POST.get("description")

        if student_name and description:
            language = detect_language(description)
            category, confidence = predict_category_with_confidence(description)
            priority = predict_priority(description)
            duplicate = detect_duplicate(description, category=category)

            departments = {
                "IT & Wi-Fi": "IT Department",
                "Hostel": "Hostel Management",
                "Academic": "Academic Office",
                "Infrastructure": "Maintenance",
                "Transport": "Transport Office"
            }

            department = departments.get(category, "Administration")

            Complaint.objects.create(
                student_name=student_name,
                description=description,
                category=category,
                priority=priority,
                department=department,
                is_duplicate=duplicate,
                language=language,
                confidence=confidence
            )

            result = {
                "category": category,
                "confidence": round(confidence * 100, 1),
                "priority": priority,
                "department": department,
                "duplicate": duplicate,
                "language": language
            }

    return render(request, "complaints/home.html", {"result": result})


# ---------------------------------------------------------------------------
# Staff dashboard view
# ---------------------------------------------------------------------------

STATUS_CHOICES = ["Open", "In Progress", "Resolved"]
CATEGORIES = ["Academic", "Hostel", "Infrastructure", "IT & Wi-Fi", "Transport"]
DEPARTMENTS = [
    "Academic Office",
    "Hostel Management",
    "IT Department",
    "Maintenance",
    "Transport Office",
    "Administration",
]


@login_required(login_url='login')
@staff_required
def dashboard(request):
    complaints = Complaint.objects.all().order_by('-created_at', '-id')

    # -- Search --
    search_query = request.GET.get('q', '').strip()
    if search_query:
        complaints = complaints.filter(description__icontains=search_query)

    # -- Filters --
    filter_category = request.GET.get('category', '')
    filter_priority = request.GET.get('priority', '')
    filter_status = request.GET.get('status', '')
    filter_department = request.GET.get('department', '')

    if filter_category:
        complaints = complaints.filter(category=filter_category)
    if filter_priority:
        complaints = complaints.filter(priority=filter_priority)
    if filter_status:
        complaints = complaints.filter(status=filter_status)
    if filter_department:
        complaints = complaints.filter(department=filter_department)

    # -- Summary counts (from unfiltered set) --
    all_complaints = Complaint.objects.all()
    summary = {
        'total': all_complaints.count(),
        'open': all_complaints.filter(status='Open').count(),
        'in_progress': all_complaints.filter(status='In Progress').count(),
        'resolved': all_complaints.filter(status='Resolved').count(),
        'high_priority': all_complaints.filter(priority='High').count(),
        'duplicates': all_complaints.filter(is_duplicate=True).count(),
    }

    context = {
        'complaints': complaints,
        'summary': summary,
        'status_choices': STATUS_CHOICES,
        'categories': CATEGORIES,
        'departments': DEPARTMENTS,
        # preserve active filter values for template
        'filter_category': filter_category,
        'filter_priority': filter_priority,
        'filter_status': filter_status,
        'filter_department': filter_department,
        'search_query': search_query,
    }
    return render(request, 'complaints/dashboard.html', context)


# ---------------------------------------------------------------------------
# Status update endpoint (POST only)
# ---------------------------------------------------------------------------

@login_required(login_url='login')
@staff_required
def update_status(request, complaint_id):
    if request.method == 'POST':
        complaint = get_object_or_404(Complaint, pk=complaint_id)
        new_status = request.POST.get('status', '').strip()
        if new_status in STATUS_CHOICES:
            complaint.status = new_status
            complaint.save()
    return redirect('dashboard')
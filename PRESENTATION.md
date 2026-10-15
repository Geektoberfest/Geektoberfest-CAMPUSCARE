# CAMPUSCARE — Presentation Guide

## Problem

Campus complaints are often submitted as unstructured text and can be difficult to categorize, prioritize and route manually.

## Proposed Solution

CAMPUSCARE provides a smart complaint intake workflow.

A student submits a complaint and the system:
1. Predicts a complaint category using TF-IDF and Logistic Regression.
2. Assigns a priority using text-based rules.
3. Routes the complaint to a department.
4. Checks whether a similar complaint already exists.
5. Stores the complaint in the database.

## Demo Flow

Use examples such as:

```text
Wi-Fi router in Block D keeps restarting and dropping connections
```
Expected language: English, category: IT & Wi-Fi, confidence: about 78%, priority: Medium.

```text
Electric shock felt from metal switchboard in physics laboratory
```
Expected language: English, category: Infrastructure, priority: High (Life-safety hazard trigger), department: Maintenance.

```text
This is not an emergency, just need information on bus fee payment
```
Expected category: Transport, priority: Low (Negation-aware: ignores 'emergency' when negated).

```text
हॉस्टल में पानी नहीं आ रहा है
```
Expected language: Hindi, category: Hostel.

```text
Hostel mein paani nahi aa raha hai
```
Expected language: Hinglish, category: Hostel.

```text
वसतिगृहात पाण्याचा पुरवठा बंद आहे
```
Expected language: Marathi, category: Hostel.

## Current Gaps

## Current Gaps

The prototype provides full multilingual complaint intake (English, Hindi, Hinglish, Marathi) with calibrated confidence scores and negation-aware priority rules. Open areas for future contributions include building a staff admin analytics dashboard, speech-to-text input, and automated email/SMS status alerts.

## Main Technical Concepts

- Django request/response flow & ORM
- Calibrated probability confidence estimation (`predict_proba`)
- Sublinear TF-IDF text representation with n-grams
- Regularized Logistic Regression with balanced class weights
- Multi-tiered negation-aware priority triage
- Cosine similarity duplicate detection
- HTML/CSS forms
- Automated Django test suite & benchmark evaluation

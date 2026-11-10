# Modern Smishing & Phishing Training Augmentation Dataset

## Overview & Purpose
This dataset (`data/smishing_augmentation.tsv`) was created to supplement the training partition of the 2011/2012 UCI SMS Spam Collection dataset. 

While the historical UCI dataset captures early-2010s premium-rate SMS, lottery scams, and ringtone clubs, it completely lacks modern smartphone phishing ("smishing") tactics that emerged over the subsequent decade.

## Scope of Augmentations

### 1. Modern Phishing / Smishing (30 samples)
- **Banking & KYC Scams**: Fake account suspension alerts, urgent KYC update demands, debit card block threats, and unauthorized transaction cancellation links.
- **Brand & Account Impersonation**: Spoofed alerts from Apple iCloud, Netflix billing, PayPal dispute center, Microsoft Office 365, and USPS/FedEx parcel delivery fees.
- **Modern Giveaway & Crypto Scams**: Fake iPhone 15 Pro giveaways, Amazon $1000 gift cards, Bitcoin/airdrop scams, and URL shortener vectors (`bit.ly`, `tinyurl.com`).
- **Urgency & Coercion Tactics**: Tight deadlines (e.g., "within 24 hours", "2 hours remaining"), threats of account termination, and fake job offers.

### 2. Balanced Legitimate Communications (31 samples)
To prevent the model from falsely associating common banking, delivery, or authentication words with spam, modern legitimate messages are included:
- **Transactional & OTP Alerts**: Real bank balance notifications, Chase/Wells Fargo 2FA OTP codes, Bank of America fraud verification questions.
- **Operational Updates**: USPS delivery confirmation, Amazon shipping notices, CVS prescription readiness, Delta flight schedules.
- **Conversational & Personal**: Casual chats, appointment reminders, DoorDash/Uber arrival notifications.

## Data Leakage Prevention & Partitioning Protocol
- **Zero Test Set Contamination**: This augmentation dataset is applied **strictly to the training partition**. The original 1,114 held-out test messages from the UCI dataset remain 100% untouched.
- **Format**: Tab-separated values (TSV) with columns `label` and `text`, compatible with the project's native ingestion pipeline.


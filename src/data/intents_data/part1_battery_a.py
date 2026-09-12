BATTERY_CASES = [
    {
        "id": "GOLD-001",
        "customer_tweet": "My iPhone 7 battery is draining from 100% to 20% in less than 2 hours without even using it. Is this normal?",
        "true_intent": "hardware_battery",
        "true_escalate": False,
        "true_escalation_reason": "Standard battery drain troubleshooting: check battery health and background app usage before hardware service.",
        "gold_reference_reply": "We want to help with your battery life! Go to Settings > Battery > Battery Health to check maximum capacity, and see which apps are using the most power. DM us if you would like to troubleshoot further.",
        "difficulty": "easy",
        "human_judge_score": 5
    },
    {
        "id": "GOLD-002",
        "customer_tweet": "My phone battery is visibly bulging and pushing the screen off the frame! What should I do??",
        "true_intent": "hardware_battery",
        "true_escalate": True,
        "true_escalation_reason": "Swollen battery is a severe hardware safety hazard. Immediate human escalation and store service required.",
        "gold_reference_reply": "Please stop using and charging your device immediately for your safety. We are escalating this to our safety team and will help you schedule an immediate Genius Bar appointment.",
        "difficulty": "easy",
        "human_judge_score": 5
    },
    {
        "id": "GOLD-003",
        "customer_tweet": "Whenever I plug my charger in, it says Accessory may not be supported and refuses to charge.",
        "true_intent": "hardware_battery",
        "true_escalate": False,
        "true_escalation_reason": "Charging port debris or cable issue. Standard troubleshooting can resolve this without human escalation.",
        "gold_reference_reply": "We can help get your device charging. Inspect the charging port for lint or debris, and try an Apple-certified cable and wall adapter. DM us your iOS version if it continues.",
        "difficulty": "easy",
        "human_judge_score": 5
    },
    {
        "id": "GOLD-004",
        "customer_tweet": "My iPhone X gets burning hot while doing basic FaceTime calls and shuts off completely.",
        "true_intent": "hardware_battery",
        "true_escalate": False,
        "true_escalation_reason": "Thermal management troubleshooting: check ambient temperature, background updates, and iOS version.",
        "gold_reference_reply": "Let us look into the temperature on your iPhone. Does this happen on Wi-Fi or cellular? Check Settings > General > Software Update to ensure you have the latest iOS. DM us with details.",
        "difficulty": "medium",
        "human_judge_score": 4
    },
    {
        "id": "GOLD-005",
        "customer_tweet": "I replaced my battery at a third party kiosk and now my phone won't turn on at all. Can you guys fix it?",
        "true_intent": "hardware_battery",
        "true_escalate": True,
        "true_escalation_reason": "Third-party unauthorized hardware tampering and device failure requires Apple Store diagnostic evaluation.",
        "gold_reference_reply": "We would be glad to look into service options for your device. Because it will not power on after service, we recommend scheduling an appointment at an Apple Store or Authorized Service Provider.",
        "difficulty": "medium",
        "human_judge_score": 5
    },
    {
        "id": "GOLD-006",
        "customer_tweet": "Phone drops from 40% to 1% instantly when I step outside into the cold weather. iPhone 6s.",
        "true_intent": "hardware_battery",
        "true_escalate": False,
        "true_escalation_reason": "Known chemical aging behavior of lithium-ion batteries in cold ambient temperatures; troubleshooting and diagnostic step.",
        "gold_reference_reply": "Cold temperatures can temporarily affect lithium-ion battery performance. Check Settings > Battery > Battery Health. If maximum capacity is below 80%, a battery replacement is recommended.",
        "difficulty": "medium",
        "human_judge_score": 5
    },
    {
        "id": "GOLD-007",
        "customer_tweet": "My Apple Watch Series 3 battery used to last 2 days, now it dies in 6 hours after updating watchOS.",
        "true_intent": "hardware_battery",
        "true_escalate": False,
        "true_escalation_reason": "Post-update background re-indexing and watchOS background sync issue. Auto-handle via unpair/re-pair troubleshooting.",
        "gold_reference_reply": "We can help with your Apple Watch battery. After an update, background indexing can take up to 48 hours. If it persists, try unpairing and re-pairing your watch via the Watch app on iPhone.",
        "difficulty": "medium",
        "human_judge_score": 4
    },
    {
        "id": "GOLD-008",
        "customer_tweet": "My iPad charger sparkled when plugged in and the port smells like burnt plastic.",
        "true_intent": "hardware_battery",
        "true_escalate": True,
        "true_escalation_reason": "Electrical burning smell / spark is an acute hardware electrical hazard requiring human escalation.",
        "gold_reference_reply": "Your safety is our top priority. Please unplug and do not use the iPad or adapter. We are escalating this to our Senior Support team immediately. Please DM us your phone number.",
        "difficulty": "easy",
        "human_judge_score": 5
    },
    {
        "id": "GOLD-009",
        "customer_tweet": "Is fast charging going to degrade my battery health faster on my iPhone 8?",
        "true_intent": "hardware_battery",
        "true_escalate": False,
        "true_escalation_reason": "Educational technical inquiry about fast charging and lithium-ion battery management.",
        "gold_reference_reply": "Great question! Apple fast charging uses built-in thermal and power regulation to charge quickly up to 80% and trickle charge after to preserve battery health safely.",
        "difficulty": "easy",
        "human_judge_score": 5
    },
    {
        "id": "GOLD-010",
        "customer_tweet": "My phone is stuck on the red battery icon with lightning cable screen and won't boot even after charging all night.",
        "true_intent": "hardware_battery",
        "true_escalate": True,
        "true_escalation_reason": "Failure to hold charge overnight indicates dead logic board power IC or depleted battery cell needing hardware service.",
        "gold_reference_reply": "We would like to help you get your phone powered up. Since it has not charged overnight, we should run remote diagnostics or set up a hardware service appointment. Please DM us your serial number.",
        "difficulty": "medium",
        "human_judge_score": 4
    },
    {
        "id": "GOLD-011",
        "customer_tweet": "Battery health shows 74% and says service recommended. How much does an official replacement cost?",
        "true_intent": "hardware_battery",
        "true_escalate": False,
        "true_escalation_reason": "Standard battery pricing and service inquiry; direct informational resolution.",
        "gold_reference_reply": "An out-of-warranty battery replacement is typically $49-$69 depending on model, or free if covered under AppleCare+. You can view exact pricing and book an appointment at support.apple.com/repair.",
        "difficulty": "easy",
        "human_judge_score": 5
    },
    {
        "id": "GOLD-012",
        "customer_tweet": "Since yesterday, Low Power Mode turns on automatically and I cannot toggle it off. It is greyed out.",
        "true_intent": "hardware_battery",
        "true_escalate": False,
        "true_escalation_reason": "Software glitch or profile constraint causing greyed out setting; resolvable via restart or settings reset.",
        "gold_reference_reply": "We can help with your Low Power Mode settings. Start by performing a force restart of your iPhone. If still greyed out, check Settings > General > VPN & Device Management for any profiles.",
        "difficulty": "medium",
        "human_judge_score": 4
    },
    {
        "id": "GOLD-013",
        "customer_tweet": "Charging only works if I hold the cable at a specific 45 degree angle. Super annoying!",
        "true_intent": "hardware_battery",
        "true_escalate": True,
        "true_escalation_reason": "Loose or physically damaged Lightning/USB-C pins inside port requiring hardware repair or professional cleaning.",
        "gold_reference_reply": "This often points to debris compacted in the Lightning port or damaged connector pins. We recommend bringing your device to an Apple Store where an advisor can safely clean or inspect the port.",
        "difficulty": "medium",
        "human_judge_score": 5
    },
    {
        "id": "GOLD-014",
        "customer_tweet": "I have been using my phone for 10 minutes and the back glass is already too hot to touch comfortably.",
        "true_intent": "hardware_battery",
        "true_escalate": False,
        "true_escalation_reason": "High CPU utilization or rogue background process; provide initial troubleshooting before escalating.",
        "gold_reference_reply": "We are here to help. Check Settings > Battery to see if a specific app is running heavily in the background. Also, ensure your iOS is updated. DM us your device model and iOS version.",
        "difficulty": "medium",
        "human_judge_score": 4
    },
    {
        "id": "GOLD-015",
        "customer_tweet": "Why does my battery percentage jump from 60% to 85% randomly when I reboot?",
        "true_intent": "hardware_battery",
        "true_escalate": False,
        "true_escalation_reason": "Battery gauge calibration issue; guide user on recalibration cycles.",
        "gold_reference_reply": "That can happen if the battery management system needs recalibration. Let the battery drain to 0%, charge uninterrupted to 100%, and perform a force restart.",
        "difficulty": "hard",
        "human_judge_score": 4
    }
]

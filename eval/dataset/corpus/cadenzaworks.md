# CadenzaWorks Employee Handbook

Version 6.0, effective Solmire 7, 2031. Stewarded by the People Studio. Each section names its keeper; please direct questions there.

## Company Overview

CadenzaWorks builds collaboration and mastering tools for music producers and recording studios. We were founded in 2018 by Lottie Verhaege and Sunil Drappel after years of trading project files over email. Our main studio-office is in the Caldermoor district of Tessenwick, with a second engineering office in Vire Hollow and an artist-relations team in the town of Brackenford.

We employ 132 people across the three offices. CadenzaWorks is independent and founder-owned; we have raised no outside money. The neon sign over the live room reads: "Keep the take, lose the friction."

Leadership is small and hands-on. Lottie Verhaege is Chief Executive. Sunil Drappel runs engineering as Chief Technology Officer. The People Studio is led by Marisol Quennec, who owns this handbook. Our Head of Data Stewardship is Orrin Halvey, who owns the Security and Data Handling section. The Head of Finance is Pell Mordaine, who approves any expense exception above the standard limits.

## Products

We offer two products with a third in early access.

### Soundbridge

Soundbridge is our flagship session-collaboration product. It lets multiple producers work on the same project with version history and stem-level comments, and it is licensed per active seat. Soundbridge is about sixty-five percent of revenue on monthly subscriptions.

### Masterloft

Masterloft is our automated mastering and loudness product. It processes finished mixes against streaming targets and is priced per processed track. Masterloft grows fastest by track volume.

### Cuepoint

Cuepoint is our early-access live-performance sync tool, in use by six touring acts. It ships only with a dedicated onboarding engineer for the first ninety-five days of any contract.

All three products send usage signals into a shared internal service we call the Metronome stream, referenced in the release process.

## Engineering On-Call Policy

Owner: Sunil Drappel.

Every product engineer joins the on-call rotation after completing two months of employment. Shifts run weekly and rotate every Monday at 12:00 local time in Tessenwick.

A shift has one primary responder and one second responder. The primary takes alerts first. If the primary does not acknowledge within twelve minutes, the alert escalates to the second. If the second does not acknowledge within a further twelve minutes, it escalates to the on-duty engineering lead on the Downbeat escalation list.

Our alerting tool is named Reverb. Reverb tracks two tiers. A Tier-Red event is a customer-facing outage with a target acknowledgement of twelve minutes and a target fix of five hours. A Tier-Amber event is a degraded service with a target acknowledgement of thirty minutes and a target fix of one business day.

On-call engineers receive a stipend of 250 chips for each full week as primary and 125 chips for each full week as second, paid in the next payroll run. An engineer paged more than three times in one overnight window may take the next day as recovery at full pay, logged under the code OVR-NIGHT.

Holiday shifts go to volunteers first. If no one volunteers twelve days before the holiday, the People Studio assigns the next engineer in order, who then earns twice the normal stipend for that week.

## Expense and Travel Policy

Owner: Pell Mordaine.

Staff may spend on reasonable business needs without prior approval up to a single-transaction cap of 450 chips. Any single expense above 450 chips needs written manager approval first. Any expense above 1,800 chips needs written approval from the Head of Finance.

Travel meals are reimbursed against a daily ceiling, not per receipt. The ceiling is 60 chips in standard cities and 90 chips in cities on the high-cost list kept by Finance. Alcohol is never reimbursable.

For ground transport we cover economy rail and standard rideshare at actual cost; premium tiers are excluded. Personal-car mileage is reimbursed at a flat 0.45 chips per kilometer.

Hotels are booked through our travel desk, with a nightly ceiling of 175 chips in standard cities and 280 chips in high-cost cities. Staff who stay with friends or family rather than a hotel may claim a flat 30 chips per night with no receipt.

Expense reports are filed in our finance system, Cashledger, within twenty-five days of the expense date. Late reports require a written exception from Pell Mordaine and are not guaranteed payment.

## Parental Leave Policy

Owner: Marisol Quennec.

CadenzaWorks gives every new parent the same leave, no matter who gave birth and no matter whether the child arrives by birth, adoption, or long-term foster placement. We call this our new-arrival benefit.

The standard entitlement is fourteen weeks of fully paid leave, which may be split into as many as three separate blocks within the first sixteen months after the child arrives. An employee must have completed five months of service before the child arrives to qualify for the full fourteen weeks; those with less than five months receive seven weeks of fully paid leave.

For the first three weeks after returning, an employee may work a reduced schedule of four days per week at full pay. This ramp-back is arranged with the manager and recorded by the People Studio.

This leave does not pause equity vesting; vesting accrues normally throughout.

## Security and Data Handling

Owner: Orrin Halvey.

All customer audio and project data is classified as Mark Vermilion. Mark Vermilion data may live only in our primary data region and may never be copied to a personal device. Access is granted per project and reviewed every quarter by the Data Stewardship team.

Internal documents are classified as Mark Granite. Mark Granite documents move freely inside the company but may never reach an external address without sign-off from the Head of Data Stewardship.

We retain customer audio data for sixty days after upload, then delete it permanently unless the customer holds the keepsafe add-on, which retains data for nine months. Access logs are kept for eighteen months regardless of retention tier.

Every employee rotates credentials every ninety days. Hardware security keys are required for all administrative access; a password alone never reaches production. A lost or stolen key must be reported to the Data Stewardship team within four hours of the employee noticing.

Any laptop leaving the country must be exchanged for a clean travel unit from the Data Stewardship team before departure. Primary laptops may never cross a border.

## Release Process

Owner: Sunil Drappel.

All three products deploy through a shared pipeline named Crossfade. Code merged to the trunk is built automatically and lands first in an internal environment called Soundcheck, where it runs against synthetic traffic for at least twelve hours.

After Soundcheck, a change moves to the Frontrow environment, which carries ten percent of live customer traffic. It must run cleanly in Frontrow for twenty-four hours with no Tier-Red and no Tier-Amber event before it can proceed.

Final release is gated by a release lead, a rotating duty held by a senior engineer for one calendar month. The lead alone may promote a change to full release and alone may order a rollback. Rollbacks are expected to finish within six minutes.

We freeze releases during the final week of the calendar year and during any week a major customer goes live for the first time. During a freeze, only Tier-Red fixes ship, and those require sign-off from both the release lead and the Chief Technology Officer.

## Working Hours and Time Off

Owner: Marisol Quennec.

We do not track hours. Each employee receives twenty-five days of paid time off per calendar year, with a three-day carryover into the following year. Days above the carryover are paid out at year end at the daily rate.

We observe nine company holidays, listed each year in our shared calendar named Setlist. Every employee also receives four floating days for any occasion, including observances not on the company list.

The Vire Hollow office closes for the first full week of Solmire each year for studio maintenance; staff there work remotely that week.

## Equipment and Workspace

Owner: Marisol Quennec.

New employees choose a laptop from an approved list at onboarding. The refresh cycle is two years. Staff may expense a home-studio setup up to a lifetime ceiling of 900 chips, covering desk, chair, monitor, and audio accessories, but not a second laptop.

Each office has a quiet floor where calls and conversation are not allowed, set aside for focused work. Rooms are booked through Setlist. The largest room in the Tessenwick office, named Liveroom, seats twenty-eight and is reserved for all-company gatherings on the second working Tuesday of each month.

## Contact and Escalation

People questions go to Marisol Quennec. Security incidents go to Orrin Halvey within the windows above. Money above the standard caps goes to Pell Mordaine. Any unresolved policy dispute is decided finally by the Chief Executive, Lottie Verhaege.

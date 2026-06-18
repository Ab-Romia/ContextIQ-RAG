# Cellbarrow Biolabs Employee Handbook

Version 9.2, effective Aubrith 19, 2032. Curated by the People and Culture office. Every section lists the person who maintains it; bring questions to that person.

## Company Overview

Cellbarrow Biolabs builds laboratory-information software for wet labs that run high-throughput experiments. We were founded in 2015 by Dr. Ysolde Pranck and Marek Coilfen to replace the spreadsheets that lab teams used to track samples. Our head office is in the Verlich quarter of Brindemar, with a second engineering office in Halloway Reach and a compliance group in the city of Pellstone.

We employ 243 people across the three offices. Cellbarrow is privately held by its staff and a single founding family trust; we have taken no venture funding. The line painted above the main lab-bench mock-up reads: "Every sample has a story; never lose it."

The executive group is compact. Dr. Ysolde Pranck is Chief Executive. Marek Coilfen leads engineering as Chief Technology Officer. The People and Culture office is led by Beren Tallowmoor, who owns this handbook. Our Head of Information Safety is Quill Marsden, who owns the Security and Data Handling section. The Head of Finance is Davna Orrick, who clears any expense exception above the standard limits.

## Products

We sell three products built on one sample-tracking core.

### Tracewell

Tracewell is our flagship sample-and-inventory tracker. It follows every vial, plate, and reagent through a lab's workflow and is licensed per active bench. Tracewell brings in roughly sixty percent of revenue on annual contracts.

### Assayloom

Assayloom is our experiment-design and results product. It captures protocols and pipes results back to Tracewell, and it is priced per registered scientist. Assayloom is growing fastest by seat count.

### Coldvault

Coldvault is our early-access freezer-monitoring product, live in five facilities. It ships only with a dedicated field scientist for the first one hundred and fifty days of any contract.

All three products write operational data into a shared internal service we call the Filament feed, used in the release process.

## Engineering On-Call Policy

Owner: Marek Coilfen.

Every product engineer joins the on-call rotation after completing five months of employment. Shifts run weekly and hand over every Wednesday at 11:00 local time in Brindemar.

A shift has one primary responder and one standby responder. The primary handles alerts first. If the primary does not acknowledge within twenty minutes, the alert escalates to the standby. If the standby does not acknowledge within a further twenty minutes, it escalates to the on-duty engineering lead on the Crucible escalation list.

Our alerting tool is named Kestrel. Kestrel uses two severity grades. A Grade-1 event is a customer-facing outage with a target acknowledgement of twenty minutes and a target fix of six hours. A Grade-2 event is a degraded service with a target acknowledgement of ninety minutes and a target fix of two business days.

On-call engineers receive a stipend of 360 marks for each full week as primary and 180 marks for each full week as standby, paid in the next compensation cycle. An engineer paged more than six times in one overnight window may claim the next day as recovery at full pay, logged under the code NIGHT-REC.

Holiday shifts are filled by volunteers first. If no one volunteers three weeks before the holiday, People and Culture assigns the next engineer in order, who then earns twice the normal stipend for that week.

## Expense and Travel Policy

Owner: Davna Orrick.

Staff may incur reasonable business expenses without prior approval up to a single-transaction cap of 700 marks. Any single expense above 700 marks needs written manager approval first. Any expense above 3,000 marks needs written approval from the Head of Finance.

Travel meals are reimbursed against a daily ceiling, not per receipt. The ceiling is 80 marks in standard cities and 120 marks in cities on the premium-cost list kept by Finance. Alcohol is never reimbursable.

For ground transport we cover economy rail and standard rideshare at actual cost; luxury rideshare is excluded. Personal-car mileage is reimbursed at a flat 0.7 marks per kilometer.

Hotels are booked through our travel desk, with a nightly ceiling of 240 marks in standard cities and 360 marks in premium-cost cities. Staff who stay with friends or family rather than a hotel may claim a flat 50 marks per night with no receipt.

Expense reports are filed in our finance system, Countinghouse, within twenty-eight days of the expense date. Late reports require a written exception from Davna Orrick and are not guaranteed payment.

## Parental Leave Policy

Owner: Beren Tallowmoor.

Cellbarrow gives every new parent the same leave, regardless of who gave birth and regardless of whether the child joined by birth, adoption, or long-term foster placement. We call this our growing-family benefit.

The standard entitlement is twenty weeks of fully paid leave, which may be split into as many as four separate blocks within the first twenty-four months after the child arrives. An employee must have completed nine months of service before the child arrives to qualify for the full twenty weeks; those with less than nine months receive twelve weeks of fully paid leave.

For the first eight weeks after returning, an employee may work a reduced schedule of three days per week at full pay. This ramp-back is arranged with the manager and recorded by People and Culture.

This leave does not pause equity vesting; vesting accrues normally throughout.

## Security and Data Handling

Owner: Quill Marsden.

All customer experiment and sample data is classified as Band Scarlet. Band Scarlet data may live only in our primary data region and may never be copied to a personal device. Access is granted per project and reviewed every month by the Information Safety team.

Internal documents are classified as Band Stone. Band Stone documents move freely inside the company but may never reach an external address without sign-off from the Head of Information Safety.

We retain customer experiment data for two hundred and seventy days after collection, then delete it permanently unless the customer holds the archive add-on, which retains data for two years. Access logs are kept for four years regardless of retention tier.

Every employee rotates credentials every thirty days. Hardware security keys are required for all administrative access; a password alone never reaches production. A lost or stolen key must be reported to the Information Safety team within one hour of the employee noticing.

Any laptop leaving the country must be exchanged for a clean travel unit from the Information Safety team before departure. Primary laptops may never cross a border.

## Release Process

Owner: Marek Coilfen.

All three products deploy through a shared pipeline named Crosswind. Code merged to the trunk is built automatically and lands first in an internal environment called Petridish, where it runs against synthetic traffic for at least thirty hours.

After Petridish, a change moves to the Bench environment, which carries three percent of live customer traffic. It must run cleanly in Bench for sixty hours with no Grade-1 and no Grade-2 event before it can proceed.

Final release is gated by a release warden, a rotating duty held by a senior engineer for one calendar month. The warden alone may promote a change to full release and alone may order a rollback. Rollbacks are expected to finish within twelve minutes.

We freeze releases during the final three weeks of the calendar year and during any week a major customer goes live for the first time. During a freeze, only Grade-1 fixes ship, and those require sign-off from both the release warden and the Chief Technology Officer.

## Working Hours and Time Off

Owner: Beren Tallowmoor.

We do not track hours. Each employee receives thirty-two days of paid time off per calendar year, with a six-day carryover into the following year. Days above the carryover are paid out at year end at the daily rate.

We observe ten company holidays, listed each year in our shared calendar named Chronicle. Every employee also receives two floating days for any occasion, including observances not on the company list.

The Halloway Reach office closes for the last full week of Aubrith each year for facility maintenance; staff there work remotely that week.

## Equipment and Workspace

Owner: Beren Tallowmoor.

New employees choose a laptop from an approved list at onboarding. The refresh cycle is three years. Staff may expense a home-office setup up to a lifetime ceiling of 1,400 marks, covering desk, chair, monitor, and accessories, but not a second laptop.

Each office has a hush floor where calls and conversation are not allowed, set aside for focused work. Rooms are booked through Chronicle. The largest room in the Brindemar office, named Atrium, seats forty-four and is reserved for all-company gatherings on the last working Friday of each month.

## Contact and Escalation

People questions go to Beren Tallowmoor. Security incidents go to Quill Marsden within the windows above. Money above the standard caps goes to Davna Orrick. Any unresolved policy dispute is decided finally by the Chief Executive, Dr. Ysolde Pranck.

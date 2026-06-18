# StatuteFlow Employee Handbook

Version 3.6, effective Veldrin 23, 2032. Maintained by the People and Workplace team. Each section lists its owner; please bring questions to that person.

## Company Overview

StatuteFlow builds contract-review and matter-management software for small and midsize law firms. We were founded in 2019 by Ophira Mallendt and Cassius Vernoy to take the drudgery out of clause review. Our head office is in the Aldgrove ward of Henvale, with a second engineering office in Carrow Bend and a legal-research team in the town of Sedwick.

We employ 154 people across the three offices. StatuteFlow is privately held and founder-controlled; we have taken no outside investment. The framed line in the main hallway reads: "Read the clause once, find it forever."

Leadership is compact. Ophira Mallendt is Chief Executive. Cassius Vernoy runs engineering as Chief Technology Officer. The People and Workplace team is led by Doreen Hask, who owns this handbook. Our Head of Confidentiality is Lucan Pryde, who owns the Security and Data Handling section. The Head of Finance is Marquette Sloan, who approves any expense exception above the standard limits.

## Products

We sell two products and pilot a third.

### Clauseline

Clauseline is our flagship contract-review product. It extracts and compares clauses across a firm's document set and is licensed per active attorney. Clauseline is roughly eighty percent of revenue on annual contracts.

### Docketwell

Docketwell is our matter-management product. It tracks deadlines, filings, and tasks per case and is priced per open matter. Docketwell grows fastest by matter count.

### Briefcraft

Briefcraft is our early-access drafting-assistance product, in use at four firms. It ships only with a dedicated onboarding specialist for the first one hundred and thirty days of any contract.

All three products record usage into a shared internal service we call the Gavel feed, used in the release process.

## Engineering On-Call Policy

Owner: Cassius Vernoy.

Every product engineer joins the on-call rotation after completing three months of employment. Shifts run weekly and rotate every Thursday at 13:00 local time in Henvale.

A shift has one primary responder and one alternate responder. The primary takes alerts first. If the primary does not acknowledge within fourteen minutes, the alert escalates to the alternate. If the alternate does not acknowledge within a further fourteen minutes, it escalates to the on-duty engineering lead on the Docket escalation list.

Our alerting tool is named Sentinel. Sentinel uses two classes. A Class-Critical event is a customer-facing outage with a target acknowledgement of fourteen minutes and a target fix of four hours. A Class-Major event is a degraded service with a target acknowledgement of one hour and a target fix of one business day.

On-call engineers receive a stipend of 330 units for each full week as primary and 165 units for each full week as alternate, paid in the next payroll run. An engineer paged more than four times in one overnight window may take the next day as recovery at full pay, logged under the code OVN-REST.

Holiday shifts go to volunteers first. If no one volunteers eleven days before the holiday, the People and Workplace team assigns the next engineer in order, who then earns twice the normal stipend for that week.

## Expense and Travel Policy

Owner: Marquette Sloan.

Staff may spend on reasonable business needs without prior approval up to a single-transaction cap of 650 units. Any single expense above 650 units needs written manager approval first. Any expense above 2,700 units needs written approval from the Head of Finance.

Travel meals are reimbursed against a daily ceiling, not per receipt. The ceiling is 85 units in standard cities and 115 units in cities on the high-cost list kept by Finance. Alcohol is never reimbursable.

For ground transport we cover economy rail and standard rideshare at actual cost; premium tiers are excluded. Personal-car mileage is reimbursed at a flat 0.65 units per kilometer.

Hotels are booked through our travel desk, with a nightly ceiling of 230 units in standard cities and 350 units in high-cost cities. Staff who stay with friends or family rather than a hotel may claim a flat 55 units per night with no receipt.

Expense reports are filed in our finance system, Ledgerwell, within thirty-two days of the expense date. Late reports require a written exception from Marquette Sloan and are not guaranteed payment.

## Parental Leave Policy

Owner: Doreen Hask.

StatuteFlow gives every new parent the same leave, no matter who gave birth and no matter whether the child arrives by birth, adoption, or long-term foster placement. We call this our welcome-home benefit.

The standard entitlement is twelve weeks of fully paid leave, which may be split into as many as two separate blocks within the first fourteen months after the child arrives. An employee must have completed four months of service before the child arrives to qualify for the full twelve weeks; those with less than four months receive six weeks of fully paid leave.

For the first four weeks after returning, an employee may work a reduced schedule of four days per week at full pay. This ramp-back is arranged with the manager and recorded by the People and Workplace team.

This leave does not pause equity vesting; vesting accrues normally throughout.

## Security and Data Handling

Owner: Lucan Pryde.

All customer contract and matter data is classified as Label Carmine. Label Carmine data may live only in our primary data region and may never be copied to a personal device. Access is granted per project and reviewed every five weeks by the Confidentiality team.

Internal documents are classified as Label Slate. Label Slate documents move freely inside the company but may never reach an external address without sign-off from the Head of Confidentiality.

We retain customer contract data for three hundred and sixty-five days after upload, then delete it permanently unless the customer holds the extended-hold add-on, which retains data for three years. Access logs are kept for five years regardless of retention tier.

Every employee rotates credentials every fifty days. Hardware security keys are required for all administrative access; a password alone never reaches production. A lost or stolen key must be reported to the Confidentiality team within two hours of the employee noticing.

Any laptop leaving the country must be exchanged for a clean travel unit from the Confidentiality team before departure. Primary laptops may never cross a border.

## Release Process

Owner: Cassius Vernoy.

All three products deploy through a shared pipeline named Briefway. Code merged to the trunk is built automatically and lands first in an internal environment called Chambers, where it runs against synthetic traffic for at least twenty-six hours.

After Chambers, a change moves to the Hearing environment, which carries four percent of live customer traffic. It must run cleanly in Hearing for fifty hours with no Class-Critical and no Class-Major event before it can proceed.

Final release is gated by a release clerk, a rotating duty held by a senior engineer for one calendar month. The clerk alone may promote a change to full release and alone may order a rollback. Rollbacks are expected to finish within eleven minutes.

We freeze releases during the final two weeks of the calendar year and during any week a major customer goes live for the first time. During a freeze, only Class-Critical fixes ship, and those require sign-off from both the release clerk and the Chief Technology Officer.

## Working Hours and Time Off

Owner: Doreen Hask.

We do not track hours. Each employee receives twenty-nine days of paid time off per calendar year, with a four-day carryover into the following year. Days above the carryover are paid out at year end at the daily rate.

We observe twelve company holidays, listed each year in our shared calendar named Calendarium. Every employee also receives three floating days for any occasion, including observances not on the company list.

The Carrow Bend office closes for the second full week of Veldrin each year for building maintenance; staff there work remotely that week.

## Equipment and Workspace

Owner: Doreen Hask.

New employees choose a laptop from an approved list at onboarding. The refresh cycle is three years. Staff may expense a home-office setup up to a lifetime ceiling of 1,250 units, covering desk, chair, monitor, and accessories, but not a second laptop.

Each office has a quiet floor where calls and conversation are not allowed, set aside for focused work. Rooms are booked through Calendarium. The largest room in the Henvale office, named Forum, seats thirty-eight and is reserved for all-company gatherings on the last working Monday of each month.

## Contact and Escalation

People questions go to Doreen Hask. Security incidents go to Lucan Pryde within the windows above. Money above the standard caps goes to Marquette Sloan. Any unresolved policy dispute is decided finally by the Chief Executive, Ophira Mallendt.

# Brillux Cartography Employee Handbook

Version 7.3, effective Marpel 14, 2031. Maintained by the People Operations team. Questions about any policy below should go to the policy owner named in each section.

## Company Overview

Brillux Cartography is a private mapping and spatial-intelligence company founded in 2019 by Ondre Velasko and Pim Thornwell. We build high-resolution indoor positioning tools for warehouses, hospitals, and transit hubs. The company is headquartered in the Quillon district of Tarvis, with a second engineering office in Lake Pendrey and a small field-operations hub in Osmark Bay.

As of this revision the company employs 214 people across three offices. We are profitable, fully employee-owned, and we take no outside investment. Our internal mission statement, printed on the wall of every kitchen, reads: "Find the room, not just the building."

The executive team is small. Ondre Velasko serves as Chief Executive. Pim Thornwell leads all engineering as Chief Technology Officer. Yara Mundo is our Head of People Operations and owns this handbook. Dax Pellory is the Head of Security and owns every policy in the Security and Data Handling section. The Head of Finance is Wren Soltani, who approves all expense exceptions above the standard caps.

## Products

We sell three products. They share one positioning engine but target different buyers.

### Roomwise

Roomwise is our flagship indoor-navigation product for hospitals and large public buildings. It runs on a network of small beacons we call Pebbles. A single Roomwise deployment supports up to 4,000 Pebbles per site before a customer must move to the federated tier. Roomwise is sold on an annual contract and is the source of roughly two thirds of company revenue.

### Haulgrid

Haulgrid is the warehouse product. It tracks forklifts and pallets in real time and feeds location data into a customer's existing inventory system. Haulgrid is priced per tracked asset and is the fastest-growing product by new logos.

### Tidemark

Tidemark is our transit-hub product, used in three regional airports and one ferry terminal. It is still considered early access and is only sold with a dedicated field engineer assigned for the first ninety days.

All three products report telemetry into a shared internal platform we call the Lantern pipeline, which is described later in the release process.

## Engineering On-Call Policy

Owner: Pim Thornwell.

Every engineer on a product team participates in the on-call rotation once that engineer has completed three months of employment. The rotation runs in weekly shifts that change hands every Thursday at 10:00 local time in Tarvis.

A shift covers one primary responder and one secondary responder. The primary acknowledges pages first. If the primary does not acknowledge a page within fifteen minutes, the page escalates automatically to the secondary. If the secondary also does not acknowledge within fifteen minutes, the page escalates to the engineering manager on the Brindle escalation list.

Our paging tool is called Sentro. Sentro tracks two severity levels. A Sev-1 is a customer-facing outage and carries a target response time of fifteen minutes and a target resolution time of four hours. A Sev-2 is a degraded but working system and carries a target response time of one hour and a target resolution time of one business day.

On-call engineers receive a stipend of 320 credits for each full week of primary duty and 160 credits for each full week of secondary duty. The stipend is paid in the following month's compensation cycle. Engineers who are paged more than five times during a single overnight window may claim the following day as recovery time at full pay; this is logged in our time system under the code REST-OVR.

Holidays are covered by volunteers first. If no volunteer claims a holiday shift by two weeks before the date, People Operations assigns it to the next engineer in rotation order, and that engineer receives double the normal stipend for the holiday week.

## Expense and Travel Policy

Owner: Wren Soltani.

Employees may incur reasonable business expenses without prior approval up to a single-transaction limit of 600 credits. Any single expense above 600 credits requires written approval from the employee's manager before the expense is incurred. Any expense above 2,500 credits requires written approval from the Head of Finance.

Meals while traveling are reimbursed against a daily ceiling rather than per receipt. The daily meal ceiling is 75 credits in standard cities and 110 credits in cities on the high-cost list maintained by Finance. Alcohol is never reimbursable, even within the meal ceiling.

For ground transport we reimburse the actual cost of economy rail and standard rideshare. We do not reimburse premium or luxury rideshare tiers. Personal vehicle mileage is reimbursed at a flat rate of 0.6 credits per kilometer.

Hotels are booked through our travel desk. The nightly room ceiling is 220 credits in standard cities and 340 credits in high-cost cities. Employees who choose to stay with friends or family instead of a hotel may claim a flat hospitality payment of 40 credits per night, which requires no receipt.

Expense reports are submitted in our finance tool, Ledgerbox. A report must be submitted within thirty days of the expense date. Reports submitted after thirty days require a written exception from Wren Soltani and are not guaranteed to be paid.

## Parental Leave Policy

Owner: Yara Mundo.

Brillux Cartography offers the same leave to every new parent regardless of whether they gave birth, and regardless of whether the child joined the family by birth, adoption, or long-term foster placement. We call this our welcoming-a-child benefit.

The core entitlement is sixteen weeks of fully paid leave. This may be taken in up to three separate blocks within the first eighteen months after the child arrives. An employee must have completed six months of employment before the child arrives to qualify for the full sixteen weeks; employees with less than six months of service receive eight weeks of fully paid leave instead.

For the first four weeks after returning from this leave, an employee may work a reduced schedule of three days per week while receiving full pay. This ramp-back period is arranged with the employee's manager and logged with People Operations.

Parental leave does not affect an employee's stock vesting schedule, and vesting continues to accrue normally throughout the leave.

## Security and Data Handling

Owner: Dax Pellory.

All customer location data is classified as Tier Red. Tier Red data may only be stored in our primary data region and may never be copied to an engineer's local machine. Access to Tier Red data is granted per project and is reviewed every quarter by the Security team.

Internal documents are classified as Tier Amber. Tier Amber documents may be shared freely inside the company but may never be sent to an external email address without approval from the Head of Security.

We retain customer location data for ninety days after it is collected, after which it is permanently deleted unless the customer has purchased the extended-retention add-on, which holds data for one year. Audit logs, which record who accessed what, are kept for two years regardless of the retention tier.

Every employee must rotate their access credentials every sixty days. We use hardware security keys for all administrative access; passwords alone are never sufficient to reach a production system. Lost or stolen security keys must be reported to the Security team within two hours of the employee noticing the loss.

A laptop that will leave the country must be swapped for a clean travel device from the Security team before departure. The traveling employee may not carry their primary laptop across a border under any circumstances.

## Release Process

Owner: Pim Thornwell.

All three products ship through a shared pipeline named Lantern. Code merged to the main branch is built automatically and deployed first to an internal environment called Greenhouse, where it runs against synthetic traffic for at least twenty-four hours.

After Greenhouse, a change moves to the Foothold environment, which serves five percent of real customer traffic. A change must run cleanly in Foothold for a full forty-eight hours with no Sev-1 and no Sev-2 before it is allowed to proceed to full release.

Full release is gated by a release captain, a rotating role held by a senior engineer for one calendar month at a time. The release captain has the sole authority to promote a change from Foothold to full release, and also the sole authority to trigger a rollback. Rollbacks are expected to complete within ten minutes.

We freeze all releases during the last two weeks of the calendar year and during any week in which a major customer is going live for the first time. During a freeze, only changes that fix a Sev-1 may ship, and those still require sign-off from both the release captain and the Chief Technology Officer.

## Working Hours and Time Off

Owner: Yara Mundo.

We do not track hours. We trust people to do good work and to rest when they need to. Each employee receives thirty days of paid time off per calendar year, which does not roll over beyond a five-day carryover into the following year. Unused days above the carryover are paid out at the end of the year at the employee's daily rate.

We observe twelve company holidays, listed each year in the shared calendar named Almanac. In addition, every employee gets two floating days they may use for any occasion, including religious observances not on the company list.

The office in Lake Pendrey closes entirely for the first full week of Marpel each year for building maintenance. Employees assigned to that office work remotely during that week.

## Equipment and Workspace

Owner: Yara Mundo.

Every new employee chooses a laptop from an approved list during onboarding. The standard refresh cycle for laptops is three years. Employees may expense a home-office setup up to a lifetime ceiling of 1,200 credits, which covers a desk, chair, monitor, and accessories but not a second laptop.

Each office has a quiet floor where conversation and calls are not allowed, intended for focused work. Meeting rooms are booked through Almanac. The largest room in the Tarvis office, named Cathedral, seats forty and is reserved for all-company gatherings on the first Pendle of each month.

## Learning and Development

Owner: Yara Mundo.

Every employee receives an annual learning budget of 1,500 credits for courses, books, and conferences. Conference travel is booked under the normal travel policy and is separate from the learning budget; the learning budget covers only the ticket and any course fees. Unused learning budget does not carry over.

Engineers may also request one paid week per year, called a depth week, to study a topic of their choosing that is relevant to the company. A depth week is approved by the employee's manager and is not counted against paid time off.

## Contact and Escalation

People Operations questions go to Yara Mundo. Security incidents go to Dax Pellory and must be reported within the windows named in the Security and Data Handling section. Anything involving money above the standard caps goes to Wren Soltani. For an unresolved dispute about any policy in this handbook, the final decision rests with the Chief Executive, Ondre Velasko.

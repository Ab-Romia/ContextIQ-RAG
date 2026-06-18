# FreightNautica Employee Handbook

Version 4.1, effective Quintar 2, 2030. Owned and kept current by the People Practices group. Each section names the colleague responsible for it; route questions to that person.

## Company Overview

FreightNautica is an ocean and inland-waterway logistics software company founded in 2017 by Halpren Yost and Cira Adelmund. We build routing, customs, and dock-scheduling tools for mid-sized shipping operators who move containers between regional ports. Our headquarters sits in the Harrowfield ward of Cestral, with a second product office in Drennan Cove and a customs-research team in the town of Mossgate.

The company currently employs 178 people across the three locations. We are bootstrapped, take no outside capital, and have been cash-flow positive since 2021. The phrase we keep on every dock-side mug reads: "A container that waits is a container that costs."

Our leadership is deliberately flat. Halpren Yost is Chief Executive. Cira Adelmund runs all of engineering as Chief Technology Officer. People Practices is led by Tomas Brevik, who owns this handbook. Our Head of Trust and Data is Inela Karsh, who owns the Security and Data Handling section. The Head of Finance is Penrose Lull, who signs off on any spending exception above the standard limits.

## Products

We sell two products today and pilot a third.

### Wakeline

Wakeline is our flagship voyage-planning product. It calculates optimal routes against fuel cost, weather, and port congestion, and it is licensed per active vessel. Wakeline accounts for roughly seventy percent of revenue and is sold on two-year contracts.

### Quaymaster

Quaymaster is our dock-scheduling tool. It assigns berths, cranes, and labor crews to arriving ships and is priced per managed berth. Quaymaster is our fastest-growing line by new customers.

### Tollwell

Tollwell is our early-access customs-clearance assistant, currently live with four operators. It ships only with a dedicated onboarding engineer for the first one hundred and twenty days of any contract.

All three products emit operational telemetry into a shared internal service we call the Beacon stream, which the release process relies on.

## Engineering On-Call Policy

Owner: Cira Adelmund.

Every product engineer joins the on-call rotation after completing four months of employment. Shifts run weekly and rotate every Tuesday at 09:00 local time in Cestral.

A shift has one primary responder and one backup responder. The primary handles incoming alerts first. If the primary does not acknowledge an alert within ten minutes, it escalates to the backup. If the backup does not acknowledge within a further ten minutes, the alert escalates to the duty engineering lead on the Halyard escalation roster.

Our alerting tool is named Pulsar. Pulsar recognizes two priority bands. A Priority-A event is a customer-facing outage with a target acknowledgement of ten minutes and a target fix of three hours. A Priority-B event is a degraded service with a target acknowledgement of forty-five minutes and a target fix of one business day.

On-call engineers earn a stipend of 280 tokens for each full week as primary and 140 tokens for each full week as backup, paid in the next payroll run. An engineer paged more than four times in a single overnight window may take the next day as recovery at full pay, logged in our time tool under the code RECOV-NT.

Holiday shifts go to volunteers first. If no one volunteers ten days before the holiday, People Practices assigns the next engineer in order, who then receives twice the normal stipend for that week.

## Expense and Travel Policy

Owner: Penrose Lull.

Staff may spend on reasonable business needs without prior sign-off up to a single-transaction cap of 500 tokens. Any single expense over 500 tokens needs written manager approval beforehand. Any expense over 2,000 tokens needs written approval from the Head of Finance.

Travel meals are reimbursed against a daily ceiling, not per receipt. The ceiling is 65 tokens in ordinary cities and 95 tokens in cities on the elevated-cost list kept by Finance. Alcohol is never reimbursable.

For ground transport we cover economy rail and standard rideshare at actual cost; premium rideshare tiers are not covered. Private-car mileage is reimbursed at a flat 0.5 tokens per kilometer.

Hotels are booked through our travel desk, with a nightly ceiling of 190 tokens in ordinary cities and 300 tokens in elevated-cost cities. Staff who lodge with friends or family rather than a hotel may claim a flat 35 tokens per night with no receipt.

Expense reports go into our finance system, Tallybridge, and must be filed within twenty-one days of the expense date. Reports filed later need a written exception from Penrose Lull and are not guaranteed payment.

## Parental Leave Policy

Owner: Tomas Brevik.

FreightNautica gives every new parent the same leave, whether or not they gave birth, and whether the child arrives by birth, adoption, or long-term foster placement. We call this our new-family benefit.

The standard entitlement is eighteen weeks of fully paid leave, which may be split into as many as two separate blocks within the first twenty months after the child arrives. An employee must have completed eight months of service before the child arrives to receive the full eighteen weeks; those with less than eight months of service receive ten weeks of fully paid leave.

For the first six weeks after returning from this leave, an employee may work a reduced schedule of four days per week at full pay. This ramp-back is arranged with the manager and recorded by People Practices.

Time on this leave does not pause equity vesting; vesting accrues normally throughout.

## Security and Data Handling

Owner: Inela Karsh.

All customer voyage and cargo data is classified as Class Crimson. Class Crimson data may live only in our primary data region and may never be copied to a personal device. Access is granted per project and reviewed every two months by the Trust team.

Internal documents are classified as Class Slate. Class Slate documents move freely inside the company but may never be sent to an external address without sign-off from the Head of Trust and Data.

We keep customer voyage data for one hundred and twenty days after collection, then delete it permanently unless the customer holds the long-hold add-on, which retains data for eighteen months. Access logs are kept for three years regardless of retention tier.

Every employee rotates access credentials every forty-five days. Hardware security keys are required for all administrative access; a password alone never reaches production. A lost or stolen key must be reported to the Trust team within ninety minutes of the employee noticing.

Any laptop leaving the country must be exchanged for a clean travel unit from the Trust team before departure. Primary laptops may never cross a border.

## Release Process

Owner: Cira Adelmund.

Both shipping products and Tollwell deploy through a shared pipeline named Driftway. Code merged to the trunk is built automatically and lands first in an internal environment called Tidepool, where it runs against simulated traffic for at least eighteen hours.

After Tidepool, a change moves to the Shoreline environment, which carries eight percent of live customer traffic. It must run cleanly in Shoreline for thirty-six hours with no Priority-A and no Priority-B event before it can proceed.

Final release is gated by a release steward, a rotating duty held by a senior engineer for one calendar month. The steward alone may promote a change to full release and alone may order a rollback. Rollbacks are expected to finish within eight minutes.

We freeze releases during the final ten days of the calendar year and during any week a major customer goes live for the first time. During a freeze, only Priority-A fixes ship, and those require sign-off from both the release steward and the Chief Technology Officer.

## Working Hours and Time Off

Owner: Tomas Brevik.

We do not track hours. Each employee receives twenty-eight days of paid time off per calendar year, with a four-day carryover into the following year. Days above the carryover are paid out at year end at the employee's daily rate.

We observe eleven company holidays, published yearly in our shared calendar named Tideboard. Every employee also receives three floating days for any occasion, including observances not on the company list.

The Drennan Cove office closes for the second full week of Quintar each year for systems maintenance; staff there work remotely that week.

## Equipment and Workspace

Owner: Tomas Brevik.

New employees pick a laptop from an approved list at onboarding. The laptop refresh cycle is four years. Staff may expense a home-office setup up to a lifetime ceiling of 1,000 tokens, covering desk, chair, monitor, and accessories, but not a second laptop.

Each office has a silence floor where calls and conversation are not allowed, reserved for deep work. Meeting rooms are booked through Tideboard. The largest room in the Cestral office, named Lighthouse, seats thirty-two and is held for all-company gatherings on the first working Monday of each month.

## Contact and Escalation

People questions go to Tomas Brevik. Security incidents go to Inela Karsh within the windows above. Money above the standard caps goes to Penrose Lull. Any unresolved policy dispute is decided finally by the Chief Executive, Halpren Yost.

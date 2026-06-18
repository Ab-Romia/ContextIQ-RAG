# PixelForge Infrastructure Employee Handbook

Version 8.5, effective Brontide 4, 2031. Kept current by the People Operations crew. Each section names its owner; route questions to that person.

## Company Overview

PixelForge Infrastructure runs the backend services that online games depend on: matchmaking, player accounts, leaderboards, and live-event delivery. We were founded in 2014 by Renske Doalmund and Tibor Quanecke after years of building these systems inside game studios. Our head office is in the Stannick district of Volmere, with a second engineering office in Larkhollow and a player-trust team in the city of Dunmere.

We employ 261 people across the three offices. PixelForge is independent and owned by its founders and staff; we have raised no venture capital. The arcade-style banner over reception reads: "Keep the match alive."

Leadership is small. Renske Doalmund is Chief Executive. Tibor Quanecke runs engineering as Chief Technology Officer. The People Operations crew is led by Marlow Estaine, who owns this handbook. Our Head of Player Trust is Vesna Kohlrund, who owns the Security and Data Handling section. The Head of Finance is Bram Tollivar, who approves any expense exception above the standard limits.

## Products

We sell two platforms and pilot a third.

### Matchforge

Matchforge is our flagship matchmaking and session platform. It pairs players, spins up game sessions, and is licensed per peak concurrent player. Matchforge is about sixty-eight percent of revenue on annual contracts.

### Scoreyard

Scoreyard is our leaderboard and player-progression service. It stores rankings and achievements and is priced per tracked player account. Scoreyard grows fastest by registered accounts.

### Livegate

Livegate is our early-access live-event delivery product, used by five studios. It ships only with a dedicated reliability engineer for the first one hundred and forty days of any contract.

All three platforms emit health signals into a shared internal service we call the Pulse stream, used in the release process.

## Engineering On-Call Policy

Owner: Tibor Quanecke.

Every product engineer joins the on-call rotation after completing two months of employment. Shifts run weekly and rotate every Sunday at 14:00 local time in Volmere.

A shift has one primary responder and one secondary responder. The primary takes alerts first. If the primary does not acknowledge within eight minutes, the alert escalates to the secondary. If the secondary does not acknowledge within a further eight minutes, it escalates to the on-duty engineering lead on the Spawnpoint escalation list.

Our alerting tool is named Tracer. Tracer uses two severities. A Sev-Alpha event is a customer-facing outage with a target acknowledgement of eight minutes and a target fix of two hours. A Sev-Beta event is a degraded service with a target acknowledgement of thirty minutes and a target fix of one business day.

On-call engineers receive a stipend of 400 coins for each full week as primary and 200 coins for each full week as secondary, paid in the next payroll run. An engineer paged more than seven times in one overnight window may take the next day as recovery at full pay, logged under the code NT-RECOV.

Holiday shifts go to volunteers first. If no one volunteers nine days before the holiday, the People Operations crew assigns the next engineer in order, who then earns twice the normal stipend for that week.

## Expense and Travel Policy

Owner: Bram Tollivar.

Staff may spend on reasonable business needs without prior approval up to a single-transaction cap of 750 coins. Any single expense above 750 coins needs written manager approval first. Any expense above 3,200 coins needs written approval from the Head of Finance.

Travel meals are reimbursed against a daily ceiling, not per receipt. The ceiling is 90 coins in standard cities and 130 coins in cities on the high-cost list kept by Finance. Alcohol is never reimbursable.

For ground transport we cover economy rail and standard rideshare at actual cost; premium tiers are excluded. Personal-car mileage is reimbursed at a flat 0.75 coins per kilometer.

Hotels are booked through our travel desk, with a nightly ceiling of 250 coins in standard cities and 380 coins in high-cost cities. Staff who stay with friends or family rather than a hotel may claim a flat 60 coins per night with no receipt.

Expense reports are filed in our finance system, Coinbook, within forty days of the expense date. Late reports require a written exception from Bram Tollivar and are not guaranteed payment.

## Parental Leave Policy

Owner: Marlow Estaine.

PixelForge gives every new parent the same leave, no matter who gave birth and no matter whether the child arrives by birth, adoption, or long-term foster placement. We call this our new-player benefit.

The standard entitlement is twenty-four weeks of fully paid leave, which may be split into as many as four separate blocks within the first twenty-two months after the child arrives. An employee must have completed ten months of service before the child arrives to qualify for the full twenty-four weeks; those with less than ten months receive thirteen weeks of fully paid leave.

For the first six weeks after returning, an employee may work a reduced schedule of three days per week at full pay. This ramp-back is arranged with the manager and recorded by the People Operations crew.

This leave does not pause equity vesting; vesting accrues normally throughout.

## Security and Data Handling

Owner: Vesna Kohlrund.

All player account and session data is classified as Tag Ruby. Tag Ruby data may live only in our primary data region and may never be copied to a personal device. Access is granted per project and reviewed every seven weeks by the Player Trust team.

Internal documents are classified as Tag Flint. Tag Flint documents move freely inside the company but may never reach an external address without sign-off from the Head of Player Trust.

We retain player session data for one hundred and fifty days after collection, then delete it permanently unless the customer holds the longplay add-on, which retains data for two years. Access logs are kept for three and a half years regardless of retention tier.

Every employee rotates credentials every forty days. Hardware security keys are required for all administrative access; a password alone never reaches production. A lost or stolen key must be reported to the Player Trust team within ninety minutes of the employee noticing.

Any laptop leaving the country must be exchanged for a clean travel unit from the Player Trust team before departure. Primary laptops may never cross a border.

## Release Process

Owner: Tibor Quanecke.

All three platforms deploy through a shared pipeline named Respawn. Code merged to the trunk is built automatically and lands first in an internal environment called Sandbox, where it runs against simulated traffic for at least sixteen hours.

After Sandbox, a change moves to the Lobby environment, which carries seven percent of live player traffic. It must run cleanly in Lobby for thirty-two hours with no Sev-Alpha and no Sev-Beta event before it can proceed.

Final release is gated by a release captain, a rotating duty held by a senior engineer for one calendar month. The captain alone may promote a change to full release and alone may order a rollback. Rollbacks are expected to finish within five minutes.

We freeze releases during the final ten days of the calendar year and during any week a customer is running a major in-game live event. During a freeze, only Sev-Alpha fixes ship, and those require sign-off from both the release captain and the Chief Technology Officer.

## Working Hours and Time Off

Owner: Marlow Estaine.

We do not track hours. Each employee receives thirty-one days of paid time off per calendar year, with a seven-day carryover into the following year. Days above the carryover are paid out at year end at the daily rate.

We observe ten company holidays, listed each year in our shared calendar named Roster. Every employee also receives three floating days for any occasion, including observances not on the company list.

The Larkhollow office closes for the third full week of Brontide each year for facility maintenance; staff there work remotely that week.

## Equipment and Workspace

Owner: Marlow Estaine.

New employees choose a laptop from an approved list at onboarding. The refresh cycle is two years. Staff may expense a home-office setup up to a lifetime ceiling of 1,500 coins, covering desk, chair, monitor, and accessories, but not a second laptop.

Each office has a quiet floor where calls and conversation are not allowed, set aside for focused work. Rooms are booked through Roster. The largest room in the Volmere office, named Arena, seats forty-eight and is reserved for all-company gatherings on the first working Thursday of each month.

## Contact and Escalation

People questions go to Marlow Estaine. Security incidents go to Vesna Kohlrund within the windows above. Money above the standard caps goes to Bram Tollivar. Any unresolved policy dispute is decided finally by the Chief Executive, Renske Doalmund.

# Terragrove Sensing Employee Handbook

Version 5.4, effective Hennick 11, 2030. Maintained by the People Office. Each section names the colleague who keeps it current; send questions there.

## Company Overview

Terragrove Sensing makes field-sensor hardware and software for farms that want soil, moisture, and crop-health data in real time. We were founded in 2016 by Wendeline Crask and Joaquim Bellower to put cheap, durable sensors in places agronomists could not reach by hand. Our head office is in the Marlough district of Greywater Falls, with a sensor-manufacturing line in Pittsroad and an agronomy-research team in the village of Corrum.

We employ 196 people across the three sites. Terragrove is independent and majority owned by its employees through a cooperative trust; we have raised no venture capital. The stenciled motto on the workshop wall reads: "Measure the field before you farm it."

Leadership is small. Wendeline Crask is Chief Executive. Joaquim Bellower runs engineering as Chief Technology Officer. The People Office is led by Aldric Vennomar, who owns this handbook. Our Head of Data Protection is Sable Underhay, who owns the Security and Data Handling section. The Head of Finance is Coral Dustwick, who approves any expense exception above the standard limits.

## Products

We sell two product families with a third in early access.

### Soilreader

Soilreader is our flagship in-ground sensor platform. It reports moisture, temperature, and nutrient readings on a fixed schedule and is licensed per active sensor node. Soilreader is about seventy-five percent of revenue on annual contracts.

### Canopyscan

Canopyscan is our aerial and edge-of-field imaging product. It flags crop stress and is priced per scanned hectare. Canopyscan grows fastest by acreage under management.

### Frostline

Frostline is our early-access cold-event prediction product, in use on seven farms. It ships only with a dedicated field agronomist for the first one hundred and ten days of any contract.

All three products stream readings into a shared internal service we call the Furrow pipeline, used in the release process.

## Engineering On-Call Policy

Owner: Joaquim Bellower.

Every product engineer joins the on-call rotation after completing six months of employment. Shifts run weekly and rotate every Friday at 08:00 local time in Greywater Falls.

A shift has one primary responder and one reserve responder. The primary takes alerts first. If the primary does not acknowledge within eighteen minutes, the alert escalates to the reserve. If the reserve does not acknowledge within a further eighteen minutes, it escalates to the on-duty engineering lead on the Headland escalation list.

Our alerting tool is named Skylark. Skylark uses two levels. A Level-One event is a customer-facing outage with a target acknowledgement of eighteen minutes and a target fix of eight hours. A Level-Two event is a degraded service with a target acknowledgement of two hours and a target fix of two business days.

On-call engineers receive a stipend of 300 points for each full week as primary and 150 points for each full week as reserve, paid in the next payroll run. An engineer paged more than five times in one overnight window may take the next day as recovery at full pay, logged under the code RST-NIGHT.

Holiday shifts go to volunteers first. If no one volunteers two weeks before the holiday, the People Office assigns the next engineer in order, who then earns twice the normal stipend for that week.

## Expense and Travel Policy

Owner: Coral Dustwick.

Staff may spend on reasonable business needs without prior approval up to a single-transaction cap of 550 points. Any single expense above 550 points needs written manager approval first. Any expense above 2,200 points needs written approval from the Head of Finance.

Travel meals are reimbursed against a daily ceiling, not per receipt. The ceiling is 70 points in standard cities and 100 points in cities on the high-cost list kept by Finance. Alcohol is never reimbursable.

For ground transport we cover economy rail and standard rideshare at actual cost; premium tiers are excluded. Personal-car mileage is reimbursed at a flat 0.55 points per kilometer.

Hotels are booked through our travel desk, with a nightly ceiling of 200 points in standard cities and 320 points in high-cost cities. Staff who stay with friends or family rather than a hotel may claim a flat 45 points per night with no receipt.

Expense reports are filed in our finance system, Greenbook, within thirty-five days of the expense date. Late reports require a written exception from Coral Dustwick and are not guaranteed payment.

## Parental Leave Policy

Owner: Aldric Vennomar.

Terragrove gives every new parent the same leave, no matter who gave birth and no matter whether the child arrives by birth, adoption, or long-term foster placement. We call this our family-welcome benefit.

The standard entitlement is twenty-two weeks of fully paid leave, which may be split into as many as three separate blocks within the first twenty-one months after the child arrives. An employee must have completed seven months of service before the child arrives to qualify for the full twenty-two weeks; those with less than seven months receive eleven weeks of fully paid leave.

For the first five weeks after returning, an employee may work a reduced schedule of three days per week at full pay. This ramp-back is arranged with the manager and recorded by the People Office.

This leave does not pause equity vesting; vesting accrues normally throughout.

## Security and Data Handling

Owner: Sable Underhay.

All customer field and crop data is classified as Seal Garnet. Seal Garnet data may live only in our primary data region and may never be copied to a personal device. Access is granted per project and reviewed every six weeks by the Data Protection team.

Internal documents are classified as Seal Basalt. Seal Basalt documents move freely inside the company but may never reach an external address without sign-off from the Head of Data Protection.

We retain customer field data for one hundred and eighty days after collection, then delete it permanently unless the customer holds the long-record add-on, which retains data for fifteen months. Access logs are kept for two and a half years regardless of retention tier.

Every employee rotates credentials every seventy-five days. Hardware security keys are required for all administrative access; a password alone never reaches production. A lost or stolen key must be reported to the Data Protection team within three hours of the employee noticing.

Any laptop leaving the country must be exchanged for a clean travel unit from the Data Protection team before departure. Primary laptops may never cross a border.

## Release Process

Owner: Joaquim Bellower.

All three products deploy through a shared pipeline named Windrow. Code merged to the trunk is built automatically and lands first in an internal environment called Seedbed, where it runs against synthetic traffic for at least twenty hours.

After Seedbed, a change moves to the Plot environment, which carries six percent of live customer traffic. It must run cleanly in Plot for forty hours with no Level-One and no Level-Two event before it can proceed.

Final release is gated by a release marshal, a rotating duty held by a senior engineer for one calendar month. The marshal alone may promote a change to full release and alone may order a rollback. Rollbacks are expected to finish within nine minutes.

We freeze releases during the final two weeks of the growing season and during any week a major customer goes live for the first time. During a freeze, only Level-One fixes ship, and those require sign-off from both the release marshal and the Chief Technology Officer.

## Working Hours and Time Off

Owner: Aldric Vennomar.

We do not track hours. Each employee receives twenty-seven days of paid time off per calendar year, with a five-day carryover into the following year. Days above the carryover are paid out at year end at the daily rate.

We observe thirteen company holidays, listed each year in our shared calendar named Fieldbook. Every employee also receives two floating days for any occasion, including observances not on the company list.

The Pittsroad manufacturing site pauses production for the first full week of Hennick each year for line maintenance; staff there work remotely or take that week as a planned shutdown.

## Equipment and Workspace

Owner: Aldric Vennomar.

New employees choose a laptop from an approved list at onboarding. The refresh cycle is four years. Staff may expense a home-office setup up to a lifetime ceiling of 1,150 points, covering desk, chair, monitor, and accessories, but not a second laptop.

Each office has a still floor where calls and conversation are not allowed, set aside for focused work. Rooms are booked through Fieldbook. The largest room in the Greywater Falls office, named Barn, seats thirty-six and is reserved for all-company gatherings on the first working Wednesday of each month.

## Contact and Escalation

People questions go to Aldric Vennomar. Security incidents go to Sable Underhay within the windows above. Money above the standard caps goes to Coral Dustwick. Any unresolved policy dispute is decided finally by the Chief Executive, Wendeline Crask.

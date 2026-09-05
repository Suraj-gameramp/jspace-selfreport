"""Natural-language "two-hop" prompt pool.

NATURAL_POOL maps each of 50 concept words to six short English descriptions
that imply the concept without naming it. Run this file as a script to verify
the hard rules (no concept words or inflections, 12-22 words, one sentence,
no exclamation marks, questions or dashes).
"""

NATURAL_POOL = {
    # ------------------------------------------------------------ countries
    "France": [
        "The country whose capital is Paris, home of the Eiffel Tower, the Louvre and Bastille Day.",
        "Croissants, berets, the Champs-Elysees and a revolution in 1789 that toppled a king.",
        "Riders in the famous July cycling race sweeping up the Champs-Elysees for the final stage in Paris.",
        "The European republic of Napoleon, Joan of Arc and Charles de Gaulle, whose tricolour flies over Paris.",
        "Bordeaux and Burgundy wine regions, the Riviera at Nice, and the Loire valley's chateaux.",
        "The nation that gave the Statue of Liberty to America and hosts the Cannes film festival.",
    ],
    "Japan": [
        "The island nation whose capital is Tokyo, land of Mount Fuji, sushi and cherry blossoms.",
        "Samurai, geisha, sumo wrestlers and bullet trains racing past rice paddies to Kyoto.",
        "Commuters in Osaka bowing politely before boarding a train that arrives to the exact second.",
        "The East Asian archipelago hit by the atomic bombs at Hiroshima and Nagasaki in 1945.",
        "Anime, manga, Nintendo, Toyota and Sony, exported from a land of shrines and karaoke bars.",
        "The country of the rising sun flag, where people write in kanji and eat with chopsticks.",
    ],
    "Brazil": [
        "The largest country in South America, whose capital is Brasilia and whose biggest city is Sao Paulo.",
        "Carnival in Rio, samba drums, the Christ the Redeemer statue and beaches at Copacabana.",
        "Barefoot children playing football on a Rio beach beneath Sugarloaf Mountain at sunset.",
        "The Portuguese-speaking nation that holds most of the Amazon rainforest and has won five World Cups.",
        "Pele, Neymar and Ayrton Senna, heroes of a football-mad country famous for coffee and carnival.",
        "The home of bossa nova, caipirinhas, the Iguazu Falls and the world's largest carnival.",
    ],
    "Egypt": [
        "The country of the pyramids at Giza, the Sphinx and the pharaohs, with its capital at Cairo.",
        "Tutankhamun's golden mask, hieroglyphs on temple walls and mummies wrapped in linen.",
        "A felucca sailing down the Nile past Luxor's temples toward the Valley of the Kings.",
        "The North African nation through which the Nile flows to the Mediterranean and the Suez Canal runs.",
        "Cleopatra's kingdom, later ruled by Nasser and Mubarak, with Alexandria on its coast.",
        "The land where Moses led the Israelites out of slavery under a pharaoh, according to Exodus.",
    ],
    "Canada": [
        "The second largest country on Earth, whose capital is Ottawa and whose flag bears a maple leaf.",
        "Mounties in red tunics, moose, ice hockey and maple syrup poured on pancakes.",
        "Fans in Toronto and Montreal packed into arenas for hockey night, the national winter obsession.",
        "The North American nation stretching from Vancouver to Newfoundland, bordering only the United States.",
        "Justin Trudeau's officially bilingual country, famous for Tim Hortons, politeness and long cold winters.",
        "The home of Niagara Falls' northern shore, the Rocky Mountain town of Banff and the Yukon.",
    ],
    "Germany": [
        "The European country whose capital is Berlin, famous for Oktoberfest, the autobahn and BMW.",
        "Bratwurst, pretzels and steins of beer, Beethoven and Bach, the Black Forest and the Rhine.",
        "Crowds with hammers chipping away at the Berlin Wall on a November night in 1989.",
        "The nation reunified in 1990 after decades split into East and West, led now from the Bundestag.",
        "Munich, Hamburg, Frankfurt and Cologne, home of Volkswagen, Mercedes and the Bundesliga.",
        "The country of Goethe, Angela Merkel and the Brothers Grimm, whose flag is black, red and gold.",
    ],
    "Norway": [
        "The Scandinavian country of deep fjords and Viking longships, whose capital is Oslo on the North Sea coast.",
        "Fjords, midnight sun, northern lights and sovereign wealth built on North Sea oil.",
        "A cross-country skier gliding through a snowy forest near Lillehammer under the northern lights.",
        "The kingdom west of Sweden that awards the Nobel Peace Prize in Oslo each December.",
        "Edvard Munch's homeland, where trolls fill the folklore and salmon fill the fjords.",
        "The country of explorers Amundsen and Nansen, of Bergen's wooden wharf and Tromso's polar nights.",
    ],
    "Chile": [
        "The country where 33 miners trapped underground were rescued live on television in 2010.",
        "Santiago, the Atacama desert, Easter Island and Patagonia, in a strip wedged between the Andes and the Pacific.",
        "Copper miners in the Atacama, pisco sours in Santiago and the poems of Pablo Neruda.",
        "The South American nation that owns Easter Island and stretches from the Atacama to Cape Horn.",
        "The country ruled by General Pinochet after the 1973 coup that overthrew Salvador Allende.",
        "The world's largest copper exporter, whose long thin territory runs down the Andes beside Argentina.",
    ],
    "Poland": [
        "The country whose capital is Warsaw, birthplace of Chopin, Copernicus and Pope John Paul II.",
        "Pierogi, Krakow's old market square, the Solidarity union and the shipyards of Gdansk.",
        "Lech Walesa addressing striking shipyard workers in Gdansk in the summer of 1980.",
        "The central European nation invaded by Hitler in September 1939, starting the Second World War.",
        "The homeland of Marie Curie, where zloty is the currency and Auschwitz stands as a memorial.",
        "The country between the Baltic coast and the Carpathian mountains, whose language uses many consonant clusters.",
    ],
    "Nepal": [
        "The Himalayan country whose capital is Kathmandu, on the south side of Everest.",
        "Sherpas, Gurkha soldiers, prayer flags and the only national flag that is not a rectangle.",
        "Trekkers setting out from Pokhara toward the Annapurna base camp past terraced hillsides.",
        "The landlocked nation between India and Tibet that contains eight of the world's ten highest peaks.",
        "The birthplace of the Buddha at Lumbini, ruled by a Hindu monarchy until 2008.",
        "Kathmandu's temples cracked and toppled by the devastating earthquake of April 2015.",
    ],
    # -------------------------------------------------------------- animals
    "spider": [
        "The eight-legged creature that spins sticky silk webs to trap flying insects.",
        "Tarantulas, black widows and the tiny house kind lurking in the corner of the bathtub.",
        "A dew-covered web strung across the garden gate at dawn, its maker waiting at the centre.",
        "The arachnid with eight legs and spinnerets that injects venom into insects caught in its web.",
        "The creature that frightened Little Miss Muffet and that Charlotte's Web made a heroine.",
        "The creepy-crawly most people fear most, though it kills flies and rarely bites humans.",
    ],
    "elephant": [
        "The largest land animal, with a trunk, ivory tusks and huge flapping ears.",
        "Matriarch-led herds trumpeting across the African savanna, spraying water over their backs.",
        "A calf at the waterhole holding its mother's tail with its trunk.",
        "The Asian and African giant poached for ivory, said to never forget and to mourn its dead.",
        "Dumbo, Babar and the animal-headed Hindu god Ganesha, all based on the same enormous creature.",
        "The beast Hannibal marched over the Alps, ridden today by mahouts in Thailand and India.",
    ],
    "tiger": [
        "The largest wild cat, striped orange and black, stalking prey in the jungles of India.",
        "Shere Khan, Tony from the cereal box and Bengal or Siberian varieties of the striped hunter.",
        "A striped hunter slipping silently through tall grass toward a deer at a jungle waterhole.",
        "The national animal of India and endangered apex predator of Asian forests, weighing up to 300 kilograms.",
        "The striped beast that William Blake described burning bright in the forests of the night.",
        "The big cat whose stripes are unique like fingerprints, reduced to a few thousand in the wild.",
    ],
    "whale": [
        "The largest animal ever to live, a marine mammal that surfaces to breathe through a blowhole.",
        "Humpbacks, sperm and killer varieties, breaching, singing and migrating across whole oceans.",
        "A boat of tourists gasping as huge tail flukes rise and slap the water off Cape Cod.",
        "The ocean giant that filters krill through baleen plates and nurses its calf on rich milk.",
        "Moby Dick, Jonah's swallower, and the creature hunted for oil from Nantucket ships.",
        "The marine mammal whose spout gives it away, protected by an international hunting moratorium since 1986.",
    ],
    "rabbit": [
        "The long-eared hopping mammal with a twitching nose that lives in burrows and nibbles carrots.",
        "Bugs Bunny, Beatrix Potter's Peter, and the white one that led Alice down a hole.",
        "A magician reaching into a top hat and pulling out a fluffy white creature by its ears.",
        "The burrowing lagomorph that breeds famously fast and lives in warrens under fields.",
        "The floppy-eared hutch pet whose foot is carried as a lucky charm.",
        "Cottontails frozen in the headlights, then bolting across a field with a flash of white tail.",
    ],
    "snake": [
        "The legless reptile that slithers, sheds its skin and flicks a forked tongue.",
        "Cobras, pythons, rattlers and vipers, some venomous, all of them swallowing their prey whole.",
        "A charmer in a Delhi market playing a flute as a hooded cobra rises from the basket.",
        "The limbless reptile with fangs and venom glands, the animal on the medical rod of Asclepius.",
        "The tempter coiled in the tree that talked Eve into eating the forbidden fruit.",
        "The scaly creature whose bite is treated with antivenom and whose rattle warns hikers.",
    ],
    "eagle": [
        "The large bird of prey with a hooked beak and talons, national emblem of the United States.",
        "Bald and golden varieties, soaring on thermals and nesting in an eyrie on a cliff.",
        "A huge raptor plunging from the sky to snatch a salmon from an Alaskan river.",
        "The raptor whose eyesight is several times sharper than a human's, hunting from great height.",
        "The bird on Roman legion standards and the American presidential seal, gripping arrows in its talons.",
        "The golf term for two under par on a hole, borrowed from a majestic raptor's name.",
    ],
    "camel": [
        "The humped desert animal that stores fat and goes for days without drinking water.",
        "Dromedaries and Bactrians, ships of the desert, carrying goods in caravans across the Sahara.",
        "A Bedouin caravan of humped beasts plodding across Saharan dunes at dusk.",
        "The one- or two-humped mammal with wide padded feet and long lashes, domesticated for desert transport.",
        "The animal in the proverb whose back is broken by a final straw.",
        "The beast whose milk, wool and hair are prized by nomads from Arabia to Mongolia.",
    ],
    "frog": [
        "The amphibian that croaks, hops and begins life as a tadpole in a pond.",
        "Kermit, the prince transformed by a kiss, and bulging eyes peering from a lily pad.",
        "A chorus croaking from the reeds on a warm spring evening after the rain.",
        "The tailless amphibian with long back legs and a sticky tongue, whose skin must stay moist.",
        "The creature dissected in school biology labs and whose legs are eaten as a delicacy in some bistros.",
        "Tiny poison dart species in Central American rainforests, brightly coloured to warn predators.",
    ],
    "owl": [
        "The nocturnal bird of prey that hoots, sees in the dark and turns its head almost fully round.",
        "Barn, snowy and tawny varieties, coughing up pellets of fur and bone after a night's hunting.",
        "A silent hunter gliding from a barn roof to seize a mouse in the moonlit field.",
        "The night-hunting raptor with forward-facing eyes, a facial disc and silent, fringed wings.",
        "Hedwig, Athena's companion and the wise old bird of children's stories, perched on a branch.",
        "The bird whose call of twit twoo carries across the woods at midnight.",
    ],
    # ---------------------------------------------------------------- foods
    "bread": [
        "The staple baked from flour, water, yeast and salt, sliced for sandwiches and toast.",
        "Baguettes, sourdough boules, bagels and sliced white loaves sold in plastic bags.",
        "A crusty loaf pulled from the oven, its inside soft and airy, torn apart while still warm.",
        "The staple made from wheat dough risen with yeast and baked, eaten in every culture on Earth.",
        "The food broken and shared in Christian communion, and whose price rise has sparked riots.",
        "Toast in the morning, a sandwich at noon, and a warm roll with dinner.",
    ],
    "cheese": [
        "The dairy product made by curdling milk with rennet and pressing and ageing the curds.",
        "Cheddar, brie, gouda and parmesan, some with holes, some with blue veins.",
        "Wheels ageing on wooden shelves in a cool cave, turned by hand for months.",
        "The fermented milk product, grated over pizza or melted in a toasted sandwich, that mice famously love.",
        "The stretchy topping on a pizza and the cubes on a cocktail stick at a wine party.",
        "The word photographers tell their subjects to say so that they smile for the camera.",
    ],
    "pasta": [
        "The Italian staple made from durum wheat, boiled and served with sauce.",
        "Spaghetti, penne, fusilli and lasagne sheets, boiled in salted water and tossed in sauce.",
        "A pot of boiling water and a nonna in Bologna rolling out tagliatelle for the ragu.",
        "The dried shapes of durum wheat semolina, cooked al dente in Italian kitchens for centuries.",
        "Carbonara, bolognese and pesto, served with grated parmesan and a glass of Chianti.",
        "The Italian staple sold dried in boxes as tubes, ribbons, bows and shells.",
    ],
    "soup": [
        "The liquid dish of simmered stock with vegetables or meat, eaten hot with a spoon from a bowl.",
        "Minestrone, gazpacho, chowder and chicken broth, ladled steaming from a big pot.",
        "A sick child propped on pillows being handed a bowl of hot chicken broth with a spoon.",
        "The first course of a formal dinner, served in a shallow bowl with a round spoon.",
        "Campbell's cans painted by Andy Warhol, and the kitchen where the homeless queue for a bowl.",
        "Tomato, lentil, pumpkin or onion, blended smooth or left chunky, with a crusty roll.",
    ],
    "chocolate": [
        "The sweet made from roasted cacao beans, sold in bars and melted for cakes.",
        "Truffles, Easter eggs, hot cocoa with marshmallows and bars of the Swiss and Belgian kind.",
        "A child licking melted cocoa sweetness from the wrapper of a bar on a hot day.",
        "The confection made from ground cacao beans, sugar and milk, toxic to dogs because of theobromine.",
        "Willy Wonka's factory, Cadbury and Hershey, and the heart-shaped boxes given on Valentine's Day.",
        "The dark, milk or white treat that melts in the mouth and stains children's fingers.",
    ],
    "honey": [
        "The golden sticky sweetener that bees make from flower nectar and store in wax combs.",
        "Jars of clover and manuka varieties, drizzled over yoghurt and stirred into tea.",
        "A beekeeper in a veil lifting a dripping frame from the hive as bees swarm around.",
        "The natural sweetener that never spoils, found still edible in ancient tombs, and fermented into mead.",
        "Winnie the Pooh's favourite food, kept in pots and stolen from the bees.",
        "The amber liquid a bear risks stings to steal from a hive.",
    ],
    "butter": [
        "The solid fat churned from cream, spread on toast and used to fry eggs.",
        "Golden pats melting on warm toast, or sizzling in a pan before the eggs go in.",
        "A farmer's wife working a wooden churn until the cream turns thick and solid.",
        "The dairy fat sold in sticks and blocks, folded into croissant dough and clarified into ghee.",
        "The salted or unsalted dairy spread that makes cinema popcorn rich and croissants flaky.",
        "The fat that goes into shortbread, pie crust and hollandaise sauce, kept in a covered dish.",
    ],
    "garlic": [
        "The pungent bulb of white cloves, crushed into sauces and famed for lingering on the breath.",
        "Papery bulbs hung in strings in the kitchen, minced and sizzled in olive oil.",
        "A cook smashing a clove with the flat of a knife and the sharp smell filling the kitchen.",
        "The allium bulb divided into cloves, used as medicine since antiquity and said to repel vampires.",
        "Aioli, tzatziki and the roasted whole bulb, its cloves turned soft and sweet.",
        "The ingredient in every Italian sauce whose smell clings to the fingers for hours.",
    ],
    "noodles": [
        "The long strands of wheat or rice dough slurped from a bowl of broth with chopsticks.",
        "Ramen, udon, soba and pho, sold from steaming street stalls across East and Southeast Asia.",
        "A student tearing open a dried block, adding the flavour sachet and boiling water for three minutes.",
        "The instant kind invented in 1958 by Momofuku Ando, now eaten by the billion each year.",
        "Stir-fried in a wok with soy sauce and vegetables, or slurped loudly as a compliment to the chef.",
        "Chow mein, lo mein and pad thai, long strands tossed with peanuts and bean sprouts.",
    ],
    "mango": [
        "The tropical stone fruit with fragrant golden flesh, the national fruit of India.",
        "Alphonso and Ataulfo varieties, sliced into a hedgehog pattern and eaten over the sink.",
        "A child in Mumbai sucking the sweet pulp from the flat stone of a ripe summer fruit.",
        "The fruit of a tropical evergreen whose skin contains the same irritant as poison ivy.",
        "Lassi, chutney, sticky rice dessert in Bangkok, all made from the same juicy tropical fruit.",
        "The golden fruit whose sticky juice runs down the chin, sold by the crate in tropical markets.",
    ],
    # --------------------------------------------------------------- colours
    "purple": [
        "The colour of amethyst, ripe plums and the robes of Roman emperors.",
        "Eggplant skin, grape juice stains and the royal dye once extracted from sea snails.",
        "A field of lavender in full bloom glowing under the late afternoon sun.",
        "The colour of royalty and Lent vestments, made historically from the costly Tyrian dye.",
        "The shade of Prince's famous rain, Barney the dinosaur and the Lakers' home jerseys.",
        "Bruises, ripe plums, irises and the rich hue of a bishop's ceremonial robe.",
    ],
    "yellow": [
        "The colour of lemons, egg yolks, sunflowers, daffodils and American school buses.",
        "Bananas, canaries, rubber ducks and the middle light on a traffic signal.",
        "A New York taxi pulling up outside a shop selling lemons and sunflowers.",
        "The colour of the sun in children's drawings, of gold, and of the legal notepads used by lawyers.",
        "Post-it notes, the Simpsons' skin, mustard and a raincoat worn in a storm.",
        "The bright hue of a ripe banana, a dandelion and a caution sign.",
    ],
    "pink": [
        "The colour of flamingos, cherry blossoms, bubblegum and a newborn's traditional blanket.",
        "Piglets, cotton candy, blushing cheeks and the ribbon for breast cancer awareness.",
        "A flock of flamingos wading through a shallow lagoon in the early morning light.",
        "The pale rosy shade of salmon flesh, ballet slippers and Barbie's signature packaging.",
        "Pepto-Bismol, a plastic flamingo lawn ornament and the cartoon Panther with the famous theme tune.",
        "The soft blush of rose petals, prawns and a girl's stereotypical nursery.",
    ],
    "brown": [
        "The colour of tree bark, garden soil, black coffee and cardboard boxes.",
        "Walnut furniture, old leather boots, acorns, roasted chestnuts and UPS delivery trucks.",
        "A muddy field after rain, with a leather saddle drying on the fence.",
        "The dark earthy colour produced by mixing all the paints together, seen in coffee and bark.",
        "Coffee, cola, cinnamon sticks, milk tea and a plain paper grocery bag.",
        "The colour of a grizzly bear, a dachshund, a monk's habit and a well-worn saddle.",
    ],
    "grey": [
        "The colour of ash, concrete, storm clouds and an old man's hair.",
        "Slate roofs, pewter tankards, steel battleships, city pigeons and a wolf's winter coat.",
        "An overcast November afternoon in a city of concrete towers and wet pavements.",
        "The neutral shade between black and white, seen in dust, gravel and granite.",
        "A dove, a mouse, a filing cabinet and a business suit for a rainy funeral.",
        "The colour of a stormy sea, silver hair, cigarette smoke and an old woollen sock.",
    ],
    "violet": [
        "The colour at the far end of the rainbow, beyond indigo, with the shortest visible wavelength.",
        "Pansies, lilacs, wisteria and the little heart-leaved woodland flower that shares its name.",
        "Wisteria dripping from a pergola in May, its blossoms the tint just beyond indigo.",
        "The gemstone tanzanite, gentian flowers and the sugared petals on old-fashioned sweets.",
        "The colour worn by suffragettes alongside white, and of the innermost band of a rainbow.",
        "The hue of iris petals, heliotrope and the flower after which it is named.",
    ],
    "beige": [
        "The pale sandy neutral of undyed wool, raw linen and old office carpets.",
        "Khaki chinos, manila envelopes, a bowl of oatmeal and a classic Burberry trench coat.",
        "A waiting room of sandy-coloured walls, tan sofas and a carpet the shade of dry sand.",
        "The bland neutral tone of putty, unbleached canvas and 1990s desktop computers.",
        "The colour of dry sand, wheat stubble, a paper coffee cup sleeve and a plain tote bag.",
        "The safest, dullest shade in the paint catalogue, chosen for rental apartments and hospital corridors.",
    ],
    "teal": [
        "The deep hue of a tropical lagoon seen from above, darker than turquoise.",
        "Vintage 1950s kitchen appliances, hospital scrubs and the Miami Dolphins' home jerseys.",
        "A row of retro diner booths upholstered in the deep aqua shade of a 1950s Chevrolet.",
        "The dark aquatic shade named for the wing patch of a small freshwater duck.",
        "The signature deep sea-toned shade of a peacock's neck feathers and the Jacksonville Jaguars' uniforms.",
        "The dark cyan shade of tropical shallows and surgical scrubs, hex code 008080 on a computer screen.",
    ],
    "navy": [
        "The dark shade of a sailor's dress uniform and a school blazer.",
        "Midnight-toned suits, pea coats and the background of the American flag's star field.",
        "A row of cadets in dark dress uniforms standing at attention on a ship's deck.",
        "The deep, almost black shade of the sea at night, named for the fleet that wore it.",
        "The classic colour of a pinstripe banker's suit, a police officer's uniform and raw selvedge denim.",
        "The darkest shade of denim jeans before they fade, and of a peacoat's heavy wool.",
    ],
    "crimson": [
        "The deep rich shade of fresh arterial blood and a cardinal's robes.",
        "Rubies, pomegranate seeds, a matador's swirling cape and Harvard University's official colour.",
        "A velvet theatre curtain falling slowly as the audience rises to applaud.",
        "The dye once made from crushed cochineal insects, prized for royal robes and lipstick.",
        "The colour of a cardinal bird, a deep rose in full bloom and Alabama's football team.",
        "Deep wine spilled on white linen, a raspberry crushed, and the robes of a Catholic cardinal.",
    ],
    # -------------------------------------------------------------- emotions
    "anger": [
        "The hot flush and clenched jaw when a driver cuts you off and then gestures rudely.",
        "Fists tightening and voice rising after catching someone in a deliberate lie.",
        "A man slamming a door so hard the frame rattles after the argument with his landlord.",
        "The surge of adrenaline that makes the face redden and the pulse pound when insulted.",
        "A fan shouting, red-faced and veins bulging, at a referee's blatant miscall in the final minute.",
        "Gritted teeth and a pounding heart at seeing someone kick a dog.",
    ],
    "sadness": [
        "The heavy, quiet low after your closest friend moves to another continent.",
        "Tears welling and a lump in the throat while watching an old family video.",
        "A woman staring out at the rain, no appetite, the house too silent since the children left.",
        "The slumped shoulders, downturned mouth and slow speech of someone whose plans have fallen through.",
        "Sitting alone on the bed with a wet face, the phone that never rings in your hand.",
        "The dull ache of a rainy Sunday after a breakup, with no energy to get dressed.",
    ],
    "guilt": [
        "Having broken a promise to a friend and being unable to look them in the eye.",
        "Lying awake replaying the harsh words you said to your mother before she went to hospital.",
        "A boy hiding the broken vase and feeling sick each time his mother mentions it.",
        "The knot in the stomach after cheating on a test and then being praised for the grade.",
        "The urge to confess and apologise that follows taking credit for a colleague's work.",
        "Eating the slice of cake you promised to save for your sister, then avoiding her all evening.",
    ],
    "shame": [
        "The burning cheeks and downcast eyes when a teacher reads your failing grade aloud.",
        "Wanting the floor to swallow you after being caught in a lie in front of everyone.",
        "A man avoiding his neighbours for weeks after his drunken scene at the street party.",
        "The head-hanging, face-hiding response to being exposed as less than you claimed to be.",
        "Being unable to meet your parents' eyes after they bail you out of jail.",
        "The urge to disappear when photographs of your worst moment are passed around the office.",
    ],
    "disgust": [
        "The wrinkled nose, curled lip and gag reflex at the smell of rotting meat.",
        "Finding a long hair in your food after you have already swallowed half the plate.",
        "A cockroach scuttling out of the cereal box onto the breakfast table.",
        "The recoil and nausea that protect us from spoiled food, faeces and open sores.",
        "Stepping barefoot in the dark onto something cold, wet and slimy on the kitchen floor.",
        "Biting into a peach and seeing half a maggot in the flesh.",
    ],
    "envy": [
        "The sour twist in the stomach when a colleague gets the promotion you wanted.",
        "Scrolling through a friend's holiday photos and resenting their luck instead of being glad.",
        "A woman forcing a smile as her sister shows off the engagement ring.",
        "The bitter wish to have what someone else has, and for them to lose it.",
        "Watching the neighbour park a new car and feeling your own suddenly look shabby.",
        "The green-eyed monster of Shakespeare, gnawing at Iago as he watches Othello's success.",
    ],
    "grief": [
        "The long, hollow sorrow that follows the death of someone deeply loved.",
        "Sorting through a dead parent's clothes, holding a jumper that still smells of them.",
        "A widow setting two places at the table out of habit, then removing one plate.",
        "The waves of crying, numbness and disbelief in the months after a funeral.",
        "Waking each morning and, after one blank second, remembering again that they are gone.",
        "The silence in the house after the mourners leave and the casseroles stop coming.",
    ],
    "boredom": [
        "The restless dullness of having nothing engaging to do for hours on end.",
        "Watching the clock crawl through the third hour of a lecture on tax law.",
        "A child in the back seat asking for the tenth time whether they are nearly there.",
        "The yawning, fidgeting stupor of a mind without stimulation, checking the phone every minute.",
        "Sitting in a waiting room with only a five-year-old magazine and a broken television.",
        "Staring at the ceiling on a wet afternoon, every game played and every show watched.",
    ],
    "dread": [
        "The sick heaviness in the stomach the night before a court date.",
        "Counting down the days to surgery with a knotted stomach and shallow breathing.",
        "An employee hearing the boss's footsteps approach, knowing the missed deadline is about to come up.",
        "The cold, sinking anticipation of something bad that is certain to come.",
        "The Sunday evening sinking feeling before returning on Monday to a job you hate.",
        "Waiting for the doctor to call with the biopsy results, flinching at every ring.",
    ],
    "despair": [
        "Having applied for two hundred jobs and been rejected by every one, with the rent overdue.",
        "The collapse when the last treatment option fails and the doctors have nothing more to offer.",
        "A farmer standing in a dust field after a third failed harvest, unable to see any way forward.",
        "The complete loss of hope in which a person stops trying, eating or getting out of bed.",
        "Staring at the ceiling at 3 a.m., certain that nothing will ever get better.",
        "The crushing sense of a refugee turned back at the last border with nowhere left to go.",
    ],
}


# ---------------------------------------------------------------- self-check
def _forbidden_words():
    """Every concept word plus plural, demonym, colour-adjective and emotion inflections."""
    words = set()
    for c in NATURAL_POOL:
        w = c.lower()
        words.update({w, w + "s", w + "es"})
    words.update({
        # demonyms / adjectives
        "french", "japanese", "brazilian", "brazilians", "egyptian", "egyptians",
        "canadian", "canadians", "german", "germans", "germanic", "norwegian",
        "norwegians", "chilean", "chileans", "polish", "poles", "pole", "nepalese",
        "nepali", "nepalis", "parisian",
        # colour adjective / variant forms
        "purplish", "yellowish", "yellowy", "pinkish", "pinky", "brownish", "browned",
        "browns", "browning", "gray", "grays", "greyish", "grayish", "greying",
        "graying", "violets", "beiges", "teals", "navies", "crimsons",
        # emotion inflections
        "angry", "angrily", "angered", "angering", "sad", "sadly", "sadder",
        "saddest", "sadden", "saddened", "guilty", "guiltily", "ashamed",
        "shameful", "shamed", "shaming", "disgusted", "disgusting", "envious",
        "enviously", "envied", "envying", "grieve", "grieved", "grieving",
        "griever", "grievous", "bore", "bores", "bored", "boring", "boringly",
        "dreaded", "dreadful", "dreading", "despairing", "despaired", "desperate",
        "desperately", "desperation",
        # animal / food inflections
        "spidery", "elephantine", "tigress", "whaling", "whaler", "whalers",
        "snakes", "snaking", "camels", "froggy", "owlet", "owlish", "breads",
        "breaded", "cheesy", "soupy", "chocolatey", "honeyed", "buttery",
        "buttered", "garlicky", "noodle", "mangoes", "mangos",
    })
    return words


def _tokens(text):
    import re
    return re.findall(r"[a-z]+", text.lower().replace("'", ""))


def run_checks():
    import re
    concepts = [c.lower() for c in NATURAL_POOL]
    forbidden = _forbidden_words()
    prefixes = [c for c in concepts if len(c) >= 4]
    violations = []

    for concept, descs in NATURAL_POOL.items():
        if len(descs) != 6:
            violations.append((concept, "<list>", f"has {len(descs)} descriptions, expected 6"))
        for d in descs:
            toks = _tokens(d)
            # rules 1 and 2: no concept words / inflections (exact or prefix match)
            for t in toks:
                if t in forbidden:
                    violations.append((concept, d, f"forbidden word '{t}'"))
                else:
                    for p in prefixes:
                        if t.startswith(p) and t != p:
                            violations.append((concept, d, f"token '{t}' begins with concept '{p}'"))
            # rule 3: 12 to 22 words, one sentence, no question
            n = len(d.split())
            if not 12 <= n <= 22:
                violations.append((concept, d, f"{n} words"))
            if "?" in d:
                violations.append((concept, d, "contains a question mark"))
            body = d.rstrip(".")
            if re.search(r"[.!?]\s", body):
                violations.append((concept, d, "more than one sentence"))
            # rule 5: no exclamation marks, no em or en dashes
            if "!" in d:
                violations.append((concept, d, "contains an exclamation mark"))
            if "—" in d or "–" in d:
                violations.append((concept, d, "contains an em or en dash"))

    if len(NATURAL_POOL) != 50:
        violations.append(("<pool>", "<pool>", f"has {len(NATURAL_POOL)} concepts, expected 50"))

    if violations:
        for concept, d, why in violations:
            print(f"[{concept}] {why}\n    {d}")
        print(f"\n{len(violations)} violation(s) found.")
    else:
        total = sum(len(v) for v in NATURAL_POOL.values())
        print(f"ALL CHECKS PASSED ({len(NATURAL_POOL)} concepts, {total} descriptions)")
    return violations


if __name__ == "__main__":
    run_checks()

# --- Auction parameters ---
AUCTION_ITEM_VALUE = 10       # True value of the auctioned item
AUCTION_MAX_BID = 15          # Maximum allowed bid
AUCTION_BID_INCREMENT = 1     # Discrete bid step size

# --- Opponent auction valuation ---
AUCTION_OPP_VALUE_LOW = 6     # Opponent's low valuation scenario
AUCTION_OPP_VALUE_HIGH = 10   # Opponent's high valuation scenario
AUCTION_OPP_DEFAULT_BID = 5   # Default opponent bid for strategies

# --- Tragedy of the Commons ---
COMMONS_RESOURCE_CAPACITY = 20    # Sustainable extraction limit
COMMONS_MAX_EXTRACTION = 10       # Max individual extraction
COMMONS_REGEN_RATE_NUM = 1        # Regeneration numerator
COMMONS_REGEN_RATE_DEN = 2        # Regeneration denominator
COMMONS_DEPLETION_PENALTY = -2    # Payoff when resource is depleted

# --- Volunteer's Dilemma ---
VOLUNTEER_BENEFIT = 6     # Benefit to all if someone volunteers
VOLUNTEER_COST = 2        # Cost to the volunteer
VOLUNTEER_NO_VOL = 0      # Payoff if nobody volunteers

# --- El Farol Bar Problem ---
EL_FAROL_CAPACITY = 6        # Bar capacity threshold
EL_FAROL_ATTEND_REWARD = 4   # Payoff for attending uncrowded bar
EL_FAROL_CROWD_PENALTY = -1  # Payoff for attending crowded bar
EL_FAROL_STAY_HOME = 2       # Payoff for staying home

# --- Generated game defaults ---
GENERATED_DEFAULT_ACTIONS = 3     # Default NxN matrix size
GENERATED_PAYOFF_MIN = -5         # Minimum random payoff
GENERATED_PAYOFF_MAX = 5          # Maximum random payoff
GENERATED_SEED_DEFAULT = 42       # Default random seed

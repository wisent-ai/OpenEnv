# --- Cournot Duopoly ---
COURNOT_DEMAND_INTERCEPT = 12   # a in P = a - b*Q
COURNOT_DEMAND_SLOPE = 1       # b in P = a - b*Q
COURNOT_MARGINAL_COST = 2      # Constant marginal cost
COURNOT_MAX_QUANTITY = 10      # Max production quantity

# --- Bertrand Competition ---
BERTRAND_MAX_PRICE = 10        # Maximum price
BERTRAND_MARGINAL_COST = 3     # Production cost
BERTRAND_MARKET_SIZE = 12      # Total demand at zero price

# --- Hotelling Location ---
HOTELLING_LINE_LENGTH = 10     # Length of the line
HOTELLING_TRANSPORT_COST = 1   # Per-unit transport cost
HOTELLING_MARKET_VALUE = 6     # Revenue per captured consumer

# --- Entry Deterrence ---
ED_MONOPOLY_PROFIT = 10        # Incumbent profit if no entry
ED_DUOPOLY_PROFIT = 4          # Each firm profit if entry and accommodate
ED_FIGHT_COST = -2             # Incumbent cost of fighting
ED_ENTRANT_FIGHT_LOSS = -3     # Entrant loss if fought
ED_STAY_OUT_PAYOFF = 0         # Entrant payoff for staying out

# --- Nash Demand Game ---
ND_SURPLUS = 10                # Total surplus to divide

# --- Double Auction ---
DA_BUYER_VALUE = 8             # Buyer private valuation
DA_SELLER_COST = 3             # Seller private cost
DA_MAX_PRICE = 10              # Maximum price

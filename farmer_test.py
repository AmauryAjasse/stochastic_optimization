import pyomo.environ as pyo

def build_model(yields):
    model = pyo.ConcreteModel()

    # Variables
    model.X = pyo.Var(["WHEAT", "CORN", "BEETS"], within=pyo.NonNegativeReals)
    model.Y = pyo.Var(["WHEAT", "CORN"], within=pyo.NonNegativeReals)
    model.W = pyo.Var(
        ["WHEAT", "CORN", "BEETS_FAVORABLE", "BEETS_UNFAVORABLE"],
        within=pyo.NonNegativeReals,
    )

    # Objective function
    model.PLANTING_COST = 150 * model.X["WHEAT"] + 230 * model.X["CORN"] + 260 * model.X["BEETS"]
    model.PURCHASE_COST = 238 * model.Y["WHEAT"] + 210 * model.Y["CORN"]
    model.SALES_REVENUE = (
        170 * model.W["WHEAT"] + 150 * model.W["CORN"]
        + 36 * model.W["BEETS_FAVORABLE"] + 10 * model.W["BEETS_UNFAVORABLE"]
    )
    model.OBJ = pyo.Objective(
        expr=model.PLANTING_COST + model.PURCHASE_COST - model.SALES_REVENUE,
        sense=pyo.minimize
    )

    # Constraints
    model.CONSTR= pyo.ConstraintList()

    model.CONSTR.add(pyo.summation(model.X) <= 500)
    model.CONSTR.add(
        yields[0] * model.X["WHEAT"] + model.Y["WHEAT"] - model.W["WHEAT"] >= 200
    )
    model.CONSTR.add(
        yields[1] * model.X["CORN"] + model.Y["CORN"] - model.W["CORN"] >= 240
    )
    model.CONSTR.add(
        yields[2] * model.X["BEETS"] - model.W["BEETS_FAVORABLE"] - model.W["BEETS_UNFAVORABLE"] >= 0
    )
    model.W["BEETS_FAVORABLE"].setub(6000)

    return model

yields = [2.5, 3, 20]
model = build_model(yields)
solver = pyo.SolverFactory("glpk")
solver.solve(model)

# Display the objective value to one decimal place
print(f"{pyo.value(model.OBJ):.1f}")
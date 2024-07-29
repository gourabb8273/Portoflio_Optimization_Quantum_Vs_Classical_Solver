from flask import Flask, jsonify, request, render_template
from flask_restful import Api, Resource
import time
import json
from dimod import Integer
import dimod
from dwave.system import EmbeddingComposite, DWaveSampler, LeapHybridCQMSampler
from ortools.linear_solver import pywraplp
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)


CLASSICAL_FILE = 'classical_result.json'
QUANTUM_FILE = 'quantum_result.json'

def save_to_file(filename, data):
    with open(filename, 'w') as file:
        json.dump(data, file)

class ClassicalOptimization(Resource):
    def get(self):
        print("Ss----------------------")
        # Define the decision variables
        solver = pywraplp.Solver.CreateSolver('GLOP')

        x1 = solver.IntVar(0, 100, 'x1')
        x2 = solver.IntVar(0, 100, 'x2')
        x3 = solver.IntVar(0, 100, 'x3')
        x4 = solver.IntVar(0, 100, 'x4')

        total_weighted_return = 0.2650 * x1 + 0.2628 * x2 + 0.2696 * x3 - 0.0088 * x4
        total_weighted_risk = 0.3013 * x1 + 0.2124 * x2 + 0.2155 * x3 + 0.4445 * x4

        solver.Add(x1 + x2 + x3 + x4 == 100)
        min_allocation_percentage = 2
        solver.Add(x1 >= min_allocation_percentage)
        solver.Add(x2 >= min_allocation_percentage)
        solver.Add(x3 >= min_allocation_percentage)
        solver.Add(x4 >= min_allocation_percentage)
        solver.Add(total_weighted_risk <= 22)

        risk_free_rate = 0.03
        excess_return = total_weighted_return - risk_free_rate
        solver.Maximize(excess_return)

        start_time = time.time()
        result_status = solver.Solve()
        classical_time = time.time() - start_time

        if result_status == pywraplp.Solver.OPTIMAL:
            x1_value = x1.solution_value()
            x2_value = x2.solution_value()
            x3_value = x3.solution_value()
            x4_value = x4.solution_value()
            final_return = 0.2650 * x1_value + 0.2628 * x2_value + 0.2696 * x3_value - 0.0088 * x4_value
            final_risk = 0.3013 * x1_value + 0.2124 * x2_value + 0.2155 * x3_value + 0.4445 * x4_value
            sharpe_ratio = final_return / final_risk

            result = {
                "return": final_return,
                "risk": final_risk,
                "sharpe_ratio": sharpe_ratio,
                "allocation": {
                    "Apple": x1_value,
                    "Microsoft": x2_value,
                    "JP Morgan": x3_value,
                    "Boeing": x4_value
                },
                "time_taken": classical_time
            }
            save_to_file(CLASSICAL_FILE, result)
            return jsonify(result)
        else:
            return jsonify({"error": "No optimal solution found"})

class QuantumOptimization(Resource):
    def get(self):
        # Define the decision variables
        stocks = {'Apple': 0, 'Microsoft': 0, 'JP Morgan': 0, 'Boeing': 0}
        x1 = Integer('x1')
        x2 = Integer('x2')
        x3 = Integer('x3')
        x4 = Integer('x4')

        total_weighted_return = 0.2650 * x1 + 0.2628 * x2 + 0.2696 * x3 - 0.0088 * x4
        total_weighted_risk = 0.3013 * x1 + 0.2124 * x2 + 0.2155 * x3 + 0.4445 * x4

        cqm = dimod.ConstrainedQuadraticModel()
        risk_free_rate = 0.03
        excess_return = total_weighted_return - risk_free_rate
        cqm.set_objective(excess_return)

        cqm.add_constraint(x1 + x2 + x3 + x4 == 100, label='total allocation')
        cqm.add_constraint(total_weighted_risk <= 22, label='risk constraint')
        min_allocation_percentage = 2
        cqm.add_constraint(x1 >= min_allocation_percentage, label='x1 allocation')
        cqm.add_constraint(x2 >= min_allocation_percentage, label='x2 allocation')
        cqm.add_constraint(x3 >= min_allocation_percentage, label='x3 allocation')
        cqm.add_constraint(x4 >= min_allocation_percentage, label='x4 allocation')

        # DWavesampler = EmbeddingComposite(DWaveSampler(solver='Advantage2_prototype2.3'))
        sampler = LeapHybridCQMSampler(token='DEV-081b757ac8fa9601e4fd99a1710375235deaa0c4')


        start_time = time.time()
        sample = sampler.sample_cqm(cqm, label="Example - Portfolio Optimization")
        quantum_time = time.time() - start_time

        energies = sample.data_vectors['energy']
        mit = max(energies)
        X_list = []
        for sample, energy, num_occ in sample.data(['sample', 'energy', 'num_occurrences']):
            if energy == mit:
                X_list.append(list(sample.values()))

        for i, stock in enumerate(stocks):
            stocks[stock] = X_list[0][i]

        final_return = 0.2650 * stocks['Apple'] + 0.2628 * stocks['Microsoft'] + 0.2696 * stocks['JP Morgan'] - 0.0088 * stocks['Boeing']
        final_risk = 0.3013 * stocks['Apple'] + 0.2124 * stocks['Microsoft'] + 0.2155 * stocks['JP Morgan'] + 0.4445 * stocks['Boeing']
        sharpe_ratio = final_return / final_risk

        result = {
            "return": final_return,
            "risk": final_risk,
            "sharpe_ratio": sharpe_ratio,
            "allocation": stocks,
            "time_taken": quantum_time
        }
        save_to_file(QUANTUM_FILE, result)
        return jsonify(result)

class Comparison(Resource):
    def get(self):
        with open(CLASSICAL_FILE, 'r') as file:
            classical_result = json.load(file)
    
        with open(QUANTUM_FILE, 'r') as file:
            quantum_result = json.load(file)
    
        return jsonify({'classical': classical_result, 'quantum': quantum_result})

@app.route('/')
def index():
    return render_template('index.html')

api.add_resource(ClassicalOptimization, '/classical')
api.add_resource(QuantumOptimization, '/quantum')
api.add_resource(Comparison, '/comparison')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

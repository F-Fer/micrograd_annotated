mod value{
    use std::collections::HashSet;
    use std::fmt;
    use std::fmt::Formatter;
    use std::hash::{Hash, Hasher};
    use std::ops::{Add, Div, Mul, Neg, Sub};
    use std::rc::Rc;
    use num::pow::Pow;

    pub struct Value{
        data: f64,  // get
        grad: f64,  // get
        backward: Box<dyn Fn()>,
        prev: HashSet<Rc<Value>>,
        op: String,
        pub label: String,
    }

    impl Value {

        // Constructors
        pub fn new(
            data: f64,
            grad: f64,
            backward: Box<dyn Fn()>,
            prev: HashSet<Rc<Value>>,
            op: String,
            label: String
        ) -> Value {
            Value {
                data,
                grad,
                backward,
                prev,
                op,
                label,
            }
        }

        pub fn from_data(
            data: f64
        ) -> Value {
            Value {
                data,
                grad: 0.0,
                backward: Box::new(|| {}),
                prev: HashSet::new(),
                op: "".to_string(),
                label: "".to_string(),
            }
        }

        pub fn from_data_prev(
            data: f64,
            prev: &HashSet<Rc<Value>>,
        ) -> Value {
            Value {
                data,
                grad: 0.0,
                backward: Box::new(|| {}),
                prev: prev.clone(),
                op: "".to_string(),
                label: "".to_string(),
            }
        }

        // Getters and setters
        pub fn get_data(&self) -> f64{
            self.data
        }

        pub fn get_grad(&self) -> f64{
            self.grad
        }

        // May not be necessary
        pub fn get_backward(&self) -> &Box<dyn Fn()>{
            &self.backward
        }

        pub fn get_prev(&self) -> &HashSet<Rc<Value>> {
            &self.prev
        }

        pub fn get_op(&self) -> &str{
            &self.op
        }

        pub fn get_label(&self) -> &str{
            &self.label
        }

        pub fn set_data(&mut self, data: f64){
            self.data = data;
        }

        pub fn set_grad(&mut self, grad: f64){
            self.grad = grad;
        }

        pub fn set_backward(&mut self, backward: Box<dyn Fn()>){
            self.backward = backward;
        }

        pub fn set_prev(&mut self, prev: HashSet<Rc<Value>>){
            self.prev = prev;
        }

        pub fn set_op(&mut self, op: String){
            self.op = op;
        }

        pub fn set_label(&mut self, label: String){
            self.label = label;
        }

        // Methods

        // Calling backward() method
        pub fn call_backward(&self){
            (self.backward)()
        }
    }

    // Display Trait
    impl fmt::Display for Value {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            write!(f, "Value({}, data: {}, grad: {})", self.label, self.data, self.grad)
        }
    }

    // Implement PartialEq
    impl PartialEq for Value {
        fn eq(&self, other: &Self) -> bool {
            // Compare all relevant fields
            self.data == other.data
                && self.grad == other.grad
                && self.op == other.op
                && self.label == other.label
                && self.prev == other.prev // Compares all Rc<Value> in the HashSet
        }
    }

    // Implement the Hash trait
    impl Hash for Value {
        fn hash<H: Hasher>(&self, state: &mut H) {
            // Hash all relevant fields
            self.data.to_bits().hash(state);
            self.grad.to_bits().hash(state);
            self.op.hash(state);
            self.label.hash(state);
            // Hash each element in the HashSet
            for item in &self.prev {
                item.hash(state);
            }
        }
    }

    // Implement Eq
    impl Eq for Value {}

    // Core operations (Add, Mul, Pow)
    impl Add for Value {
        type Output = Value;
        fn add(self, other: Value) -> Value {
            let data = self.data + other.data;

            // Creating the Set of children
            let mut prev = HashSet::new();
            prev.insert(self);
            prev.insert(other);

            let mut val = Value {
                data: data,
                grad: 0.0,
                backward: Box::new(()),
                prev: *prev,
                op: "+".to_string(),
                label: "".to_string()
            };

            // Defining the backwards function
            fn backward(mut a: &Value, mut b: &Value, out: &Value){
                let grad_self = (a.get_grad() + b.get_grad()) * out.get_grad();
                let grad_other = (b.get_grad() + a.get_grad()) * out.get_grad();
                a.set_grad(grad_self);
                b.set_grad(grad_other);
            }
            val.set_backward(Box::new(|| backward(&self, &other, &val)));

            val
        }
    }

    impl Mul for Value {
        type Output = Value;
        fn mul(self, other: Value) -> Value {
            let data = self.data * other.data;

            // Creating the Set of children
            let mut prev = HashSet::new();
            prev.insert(self);
            prev.insert(other);

            let mut val = Value {
                data: data,
                grad: 0.0,
                backward: Box::new(()),
                prev: *prev,
                op: "*".to_string(),
                label: "".to_string()
            };

            // Defining the backwards function
            fn backward(mut a: &Value, mut b: &Value, out: &Value){
                let grad_self = b.get_grad() * out.get_grad();
                let grad_other = a.get_grad() * out.get_grad();
                a.set_grad(grad_self);
                b.set_grad(grad_other);
            }
            val.set_backward(Box::new(|| backward(&self, &other, &val)));

            val
        }
    }

    impl Value {
        // Only allowing a^b with b i32
        pub fn ipow(self, exp: i32) -> Value {
            let data = self.data.powi(exp);

            // Creating the Set of children
            let mut prev = HashSet::new();
            prev.insert(self);

            let mut val = Value {
                data: data,
                grad: 0.0,
                backward: Box::new(()),
                prev: *prev,
                op: "exp".to_string(),
                label: "".to_string()
            };

            // Defining the backwards function
            fn backward(mut a: &Value, exp: i32, out: &Value){
                let grad_self = a.get_grad() + ((exp as f64) * a.get_data().powi(exp - 1)) * out.get_grad();
                a.set_grad(grad_self);
            }
            val.set_backward(Box::new(|| backward(&self, *exp, &val)));

            val
        }
    }

    impl Neg for Value {
        type Output = Value;
        fn neg(self) -> Self::Output {
            self * (-1.0)
        }
    }

    // Additional operations (Neg, Sub, Div)
    impl Sub for Value {
        type Output = Value;
        fn sub(self, other: Value) -> Value {
            self + (-other)
        }
    }

    impl Div for Value {
        type Output = Value;
        fn div(self, other: Value) -> Value {
            self * other.ipow(-1)
        }
    }

    // Operations with "other" parameters not being of Value
    impl Add<i32> for Value {
        type Output = Value;
        fn add(self, other: i32) -> Value {
            let b = Value::from_data(other as f64);
            self + b
        }
    }

    impl Add<f64> for Value {
        type Output = Value;
        fn add(self, other: f64) -> Value {
            let b = Value::from_data(other);
            self + b
        }
    }

    impl Mul<i32> for Value {
        type Output = Value;
        fn mul(self, other: i32) -> Value {
            let b = Value::from_data(other as f64);
            self * b
        }
    }

    impl Mul<f64> for Value {
        type Output = Value;
        fn mul(self, other: f64) -> Value {
            let b = Value::from_data(other);
            self * b
        }
    }

}
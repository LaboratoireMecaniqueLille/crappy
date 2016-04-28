Condition objects
=================

With the Blocks and Links, we already have a working framework to design and 
custom tests. But if you need to do something slightly different than what is 
implemented in the Blocks, you have to re-write the whole thing.

To bring some flexibility to the whole system, we added conditions. They are 
small classes, implemented by the user, and they are added on the links.
A link paired with a condition won't simply transfer data, but transfer the 
result of the data passed by the condition.
That mean almost infinite possibilites :

- Modify the values you are passing
- Evaluate a composition of the values
- Decide wether you should transfer - or not - the data
- Transfer something completely different from the input values

This can be used to send a signal to another block (for example to synchronise 
a camera), filter the signal with a mean, include numerical modelisation in 
real-time in your test...
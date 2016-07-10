module Main where

import Lib
import Data.List (foldl')
import System.Random (randomRs, getStdGen)

main :: IO ()
main = do
    putStrLn "This is sample of learning algorithm with single neuron."
    g <- getStdGen
    let
        ws_with_theta = take 3 $ randomRs (0.0, 0.01) g :: [Double]
        xyss = [[0,1,0],[1,0,0],[0,0,0],[1,3,1],[2,1,1],[1.5,2,1]] :: [[Double]]
        xyss_with_theta = map (\xs -> 1:xs) xyss
    learned_ws <- learn_IO ws_with_theta xyss_with_theta
    putStrLn $ "Training data: " ++ show xyss
    putStrLn $ "initial weights: " ++ (show ws_with_theta)
    putStrLn $ "Learned weights: " ++ show learned_ws
    learned_output learned_ws xyss

-- positive constant for updating each weight
epsilon :: Double
epsilon = 0.5

-- Parameter of sigmoid function (gain)
alpha :: Double
alpha = 4

-- Sigmoid function
sigmoid :: Floating a => a -> a -> a
sigmoid alpha s = 1 / (1 + exp (-alpha * s))

-- Single neuron
neuron :: [Double] -> [Double] -> Double
neuron ws xs = sigmoid alpha $ sum $ zipWith (+) ws xs

-- Cost function for sequential learning
-- ws: weights
-- xys: inputs and output (one data)
cost :: [Double] -> [Double] -> Double
cost ws xys = (y - y_hat)^2
    where
        y = last xys
        y_hat = neuron ws $ init xys

-- Differential of cost function (slope)
-- ws: weights
-- xys: inputs and output (one data)
diff_cost :: [Double] -> [Double] -> Int -> Double
diff_cost ws xys n = alpha * y_hat * (y - y_hat) * (y_hat - 1.0) * (xys !! n)
    where
        y = last xys
        y_hat = neuron ws $ init xys

-- Update each weight
-- ws: weights
-- xys: inputs and output (one data)
update_ws :: [Double] -> [Double] -> [Double]
update_ws ws xys = map (\n -> ws !! n - epsilon * slope n) [0..(length ws - 1)]
    where
        slope = diff_cost ws xys

-- 学習データを一周する
-- ws: weights
-- xyss: inputs and output (all data)
update_1_cycle :: [Double] -> [[Double]] -> [Double]
update_1_cycle ws [] = ws
update_1_cycle ws (xys:xyss) = update_1_cycle ws' xyss
    where ws' = update_ws ws xys

learn :: [Double] -> [[Double]] -> [Double]
learn ws xyss =
    let
        ws' = update_1_cycle ws xyss
        cost_old = sum $ map (cost ws) xyss
        cost_new = sum $ map (cost ws') xyss
    in
        if cost_old <= cost_new
            then ws
            else learn ws' xyss

learn_IO :: [Double] -> [[Double]] -> IO [Double]
learn_IO ws xyss = do
    let
        ws' = update_1_cycle ws xyss
        cost_old = sum $ map (cost ws) xyss
        cost_new = sum $ map (cost ws') xyss
    --putStrLn $ "weights: " ++ show ws
    --putStrLn $ "cost: " ++ show cost_old
    if cost_old <= cost_new
        then return ws
        else learn_IO ws' xyss

-- print outputs from neuron with learned weights
learned_output _  [] = return ()
learned_output ws (xys:xyss) = do
    let
        xs = init xys
        y = last xys
    putStrLn $ "Input: " ++ show xs
    putStrLn $ "Target output: " ++ show y
    putStrLn $ "Output of neuron: " ++ (show $ neuron ws xs) ++ "\n"
    learned_output ws xyss